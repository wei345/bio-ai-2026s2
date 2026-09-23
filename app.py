import os
import json
import glob
import shutil
from datetime import datetime
from enum import Enum
from dataclasses import asdict
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash

# Force headless mode for matplotlib before importing the analysis pipeline
os.environ['HEADLESS_MODE'] = '1'

# Import your analysis pipeline
from smash_analysis import (
    extract_kinematic_metrics, find_smashes, assess_smashes,
    plot_smashes_kinematics, overlay_kinematic_assessment,
    FrameMetrics, SmashEvent, SmashAssessment, RiskLevel,
    format_sequence
)

app = Flask(__name__)
app.secret_key = "super_secret_key" # Required for flash messages
ANALYSES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyses')
os.makedirs(ANALYSES_DIR, exist_ok=True)

# region JSON Serialization Helpers
class KinematicsEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

def save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=KinematicsEncoder, indent=2)

def load_json(filepath, cls):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [cls(**item) for item in data]
    return cls(**data)
# endregion

# Register the function globally so Jinja can use it in the HTML templates
app.jinja_env.globals.update(format_sequence=format_sequence)

# region HTML Templates (Bootstrap 5)
HTML_TOP = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Badminton Kinematics AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .navbar { margin-bottom: 2rem; }
        
        /* Interactive Row Styles */
        .row-toggle { cursor: pointer; transition: background-color 0.2s; }
        .row-toggle:hover { background-color: #f1f3f5; }
        .toggle-icon { transition: transform 0.3s ease; display: inline-block; }
        .row-toggle[aria-expanded="true"] .toggle-icon { transform: rotate(180deg); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Badminton Kinematics AI</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/analyses">Analyses</a>
                <a class="nav-link" href="/analyses/new">New Analysis</a>
            </div>
        </div>
    </nav>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="alert alert-{{ category if category != 'error' else 'danger' }}">{{ message }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}
"""

HTML_BOTTOM = """
    </div>
    <!-- Include Bootstrap JS for collapsible components -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Helper Script for seeking video times and setting playback speed -->
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            var videos = document.querySelectorAll('video');
            videos.forEach(function(vid) {
                vid.defaultPlaybackRate = 0.25;
                vid.playbackRate = 0.25;
            });
        });

        function seekVideo(timeInSeconds) {
            var vOrig = document.getElementById('origVideo');
            var vOver = document.getElementById('overlayVideo');
            
            // Sync times, reset speed, and ensure they do not auto-play
            if(vOrig) {
                vOrig.currentTime = timeInSeconds;
                vOrig.playbackRate = 0.25;
                vOrig.pause(); 
            }
            if(vOver) {
                vOver.currentTime = timeInSeconds;
                vOver.playbackRate = 0.25;
                vOver.pause();
            }
            
            // Scroll the video container smoothly into view
            // var container = document.getElementById('videoContainer');
            // if(container) {
            //     container.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // }
        }
        
        
        // Listen globally for Bootstrap trying to show or hide the collapse panel
        document.addEventListener('show.bs.collapse', function (event) {
            // Check if the element that triggered the click has 'seekVideo' in its onclick
            const trigger = event.relatedTarget || document.activeElement;
            
            if (trigger && trigger.closest('a[onclick*="seekVideo"]')) {
                event.preventDefault();
            }
        });
        
        document.addEventListener('hide.bs.collapse', function (event) {
            const trigger = event.relatedTarget || document.activeElement;
            
            if (trigger && trigger.closest('a[onclick*="seekVideo"]')) {
                event.preventDefault();
            }
        });

    </script>
</body>
</html>
"""

HOME_HTML = HTML_TOP + """
<div class="p-5 mb-4 bg-white rounded-3 shadow-sm border">
    <div class="container-fluid py-5">
        <h1 class="display-5 fw-bold">Badminton Smash Assessment Tool</h1>
        <p class="col-md-8 fs-4">An automated AI pipeline using computer vision to extract 2D kinematics, detect overhead smashes, and evaluate critical shoulder injury risk factors: proximal-to-distal sequencing, velocity amplification, and upper-arm deceleration.</p>
        <a href="/analyses" class="btn btn-primary btn-lg">View Analyses</a>
        <a href="/analyses/new" class="btn btn-outline-secondary btn-lg">Upload New Video</a>
    </div>
</div>
""" + HTML_BOTTOM

LIST_HTML = HTML_TOP + """
<div class="d-flex justify-content-between align-items-center mb-4">
    <h2>Analysis Instances</h2>
    <a href="/analyses/new" class="btn btn-success">Create New Analysis</a>
</div>
<div class="card shadow-sm">
    <table class="table table-hover mb-0">
        <thead class="table-light">
            <tr>
                <th>Instance ID</th>
                <th>Note</th>
                <th>Status</th>
                <th class="text-end">Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for instance in instances %}
            <tr>
                <td class="align-middle"><strong>{{ instance.id }}</strong></td>
                <td class="align-middle">{{ instance.note }}</td>
                <td class="align-middle">
                    {% if instance.has_assessment %}
                        <span class="badge bg-success">Assessed</span>
                    {% elif instance.has_analysis %}
                        <span class="badge bg-info text-dark">Analyzed</span>
                    {% else %}
                        <span class="badge bg-warning text-dark">Uploaded (Pending Analysis)</span>
                    {% endif %}
                </td>
                <td class="text-end">
                    <a href="/analyses/{{ instance.id }}" class="btn btn-sm btn-primary">View / Process</a>
                    <form action="/analyses/{{ instance.id }}/delete" method="POST" class="d-inline">
                        <button type="submit" class="btn btn-sm btn-danger" onclick="return confirm('Delete this instance forever?');">Delete</button>
                    </form>
                </td>
            </tr>
            {% else %}
            <tr><td colspan="4" class="text-center py-4">No analyses found.</td></tr>
            {% endfor %}
        </tbody>
    </table>
</div>
""" + HTML_BOTTOM

NEW_HTML = HTML_TOP + """
<div class="row justify-content-center">
    <div class="col-md-8">
        <div class="card shadow-sm">
            <div class="card-header bg-white"><h4 class="mb-0">Upload New Video</h4></div>
            <div class="card-body">
                <form action="/analyses/new" method="POST" enctype="multipart/form-data" onsubmit="document.getElementById('uploadBtn').disabled=true; document.getElementById('uploadBtn').innerHTML='Uploading...'; return true;">
                    <div class="mb-3">
                        <label class="form-label">Video File (.mp4, .mov)</label>
                        <input class="form-control" type="file" name="video" accept="video/*" onchange="document.getElementById('noteInput').value = this.files[0].name" required>
                    </div>
                    <div class="mb-4">
                        <label class="form-label">Note (Optional)</label>
                        <input type="text" class="form-control" id="noteInput" name="note" placeholder="e.g., Player A - Baseline Smashes">
                    </div>
                    <button type="submit" id="uploadBtn" class="btn btn-primary w-100">Upload Video</button>
                </form>
            </div>
        </div>
    </div>
</div>
""" + HTML_BOTTOM

DETAIL_HTML = HTML_TOP + """
<div class="d-flex justify-content-between align-items-center mb-4">
    <h2>Analysis: {{ instance_id }}</h2>
    <a href="/analyses" class="btn btn-outline-secondary btn-sm">Back to List</a>
</div>

<div class="row">
    
    <!-- Left Column: Smash Analysis Processing Panel -->
    <div class="col-md-6 mb-4">
        <div class="card shadow-sm h-100 border-info"> <!-- Added h-100 to equalize heights -->
            <div class="card-body d-flex flex-column justify-content-between"> <!-- Flexbox alignment -->
                <div>
                    <h5 class="card-title">Smash Analysis</h5>
                    <p class="text-muted small mb-0">Extract kinematics and identify smash timeframes.</p>
                </div>
                
                {% if not has_analysis %}
                <form action="/analyses/{{ instance_id }}/analyze" method="POST" class="mt-3" onsubmit="document.getElementById('analyzeBtn').disabled=true; document.getElementById('analyzeBtn').innerHTML='Processing...'; return true;">
                    <button type="submit" id="analyzeBtn" class="btn btn-info w-100 text-white">Analyze Smashes</button>
                </form>
                {% else %}
                <div class="mt-3 mb-0 py-2">
                    Found <strong>{{ smashes|length }}</strong> smashes.
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Right Column: Assessment Control Panel -->
    <div class="col-md-6 mb-4">
        <div class="card shadow-sm h-100 border-primary">
            <div class="card-body">
                <h5 class="card-title">Run Injury Risk Assessment</h5>
                <form action="/analyses/{{ instance_id }}" method="POST" class="row g-3 align-items-end" onsubmit="document.getElementById('assessBtn').disabled=true; document.getElementById('assessBtn').innerHTML='Processing...'; return true;">
                    
                    <!-- Expanded from col-md-7 to col-md-9 to make the input row much longer -->
                    <div class="col-md-12">
                        <label class="form-label">Max Critical Deceleration (deg/s²)</label>
                        <div class="input-group">
                            <div class="input-group-text">
                                <input class="form-check-input mt-0" type="radio" name="dec_option" value="auto" {% if saved_dec_option == 'auto' %}checked{% endif %} {% if not has_analysis %}disabled{% endif %}>
                                <span class="ms-2">Auto (Max: {{ auto_max_dec|round|int }})</span>
                            </div>
                            <div class="input-group-text">
                                <input class="form-check-input mt-0" type="radio" name="dec_option" value="custom" {% if saved_dec_option == 'custom' %}checked{% endif %} {% if not has_analysis %}disabled{% endif %}>
                                <span class="ms-2">Custom</span>
                            </div>
                            <!-- This input now has maximum horizontal space to expand into -->
                            <input type="number" class="form-control" name="custom_dec" value="{{ saved_custom_dec }}" placeholder="e.g. 15000" {% if not has_analysis %}disabled{% endif %}>
                        </div>
                    </div>
                    
                    <div class="col-md-8">
                        <button type="submit" id="assessBtn" class="btn btn-primary w-100" {% if not has_analysis %}disabled{% endif %}>Generate Assessment</button>
                    </div>
                </form>
                {% if not has_analysis %}
                <small class="text-danger mt-2 d-block">Please run Smash Analysis first.</small>
                {% endif %}
            </div>
        </div>
    </div>


</div>


{% if has_analysis %}
<!-- Primary Video Panel -->
<div class="row mb-4" id="videoContainer">
    <div class="col-12">
        <div class="card shadow-sm h-100">
            <div class="card-header bg-white d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Video</h5>
                {% if has_assessment %}
                <div class="btn-group" role="group">
                    <input type="radio" class="btn-check" name="videoToggle" id="origVidBtn" autocomplete="off" onchange="document.getElementById('origVideo').style.display='block'; document.getElementById('overlayVideo').style.display='none';">
                    <label class="btn btn-outline-primary btn-sm" for="origVidBtn">Original Video</label>

                    <input type="radio" class="btn-check" name="videoToggle" id="overlayVidBtn" autocomplete="off" checked onchange="document.getElementById('origVideo').style.display='none'; document.getElementById('overlayVideo').style.display='block';">
                    <label class="btn btn-outline-primary btn-sm" for="overlayVidBtn">Overlay Video</label>
                </div>
                {% endif %}
            </div>
            <div class="card-body p-0 bg-dark text-center">
                <video id="origVideo" src="/analyses/{{ instance_id }}/{{ input_filename }}" controls class="w-100" style="{% if has_assessment %}display: none;{% endif %}"></video>
                {% if has_assessment %}
                <video id="overlayVideo" src="/analyses/{{ instance_id }}/analyzed.mp4" controls class="w-100" style="display: block;"></video>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<!-- Smashes & Assessments Table -->
<div class="row mb-4">
    <div class="col-12">
        <div class="card shadow-sm">
            <div class="card-header bg-white"><h5 class="mb-0">Smashes</h5></div>
            <table class="table table-hover mb-0">
                <thead class="table-light">
                    <tr>
                        <th>#</th>
                        <th>Time Window</th>
                        <th>P-D Seq / Risk</th>
                        <th>V-Amp / Risk</th>
                        <th>UA Decel (deg/s<sup>2</sup>) / Risk</th>
                        <th>Overall Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {% for s in smashes %}
                    <tr {% if has_assessment %}class="row-toggle" data-bs-toggle="collapse" data-bs-target="#plot-{{ loop.index }}" aria-expanded="false"{% endif %}>
                        <td class="align-middle">{{ loop.index }}</td>
                        <td class="align-middle">
                            <a href="javascript:void(0);" onclick="seekVideo({{ s.start_frame_idx / fps }})" class="video-seeker text-decoration-none fw-bold" title="Click to view in video">
                                {{ s.start_time_str[:5] }} - {{ s.end_time_str[:5] }}
                            </a>
                        </td>
                        
                        {% if has_assessment %}
                            {% set a = assessments[loop.index0] %}
                            <td class="align-middle {% if a.p_d_sequence_risk == 0 %}{% elif a.p_d_sequence_risk == 1 %}text-warning{% else %}text-danger{% endif %}">
                                {{ format_sequence(a.p_d_sequence) }}
                            </td>
                            <td class="align-middle {% if a.velocity_amplification_risk == 0 %}{% elif a.velocity_amplification_risk == 1 %}text-warning{% else %}text-danger{% endif %}">
                                {{ format_sequence(a.velocity_amplification) }}
                            </td>
                            <td class="align-middle {% if a.critical_deceleration_risk == 0 %}{% elif a.critical_deceleration_risk == 1 %}text-warning{% else %}text-danger{% endif %}">
                                {{ a.critical_deceleration|abs|round|int }}
                            </td>
                            <td class="align-middle d-flex justify-content-between align-items-center">
                                <div>
                                    {% if a.overall_risk == 0 %}<span class="badge bg-success">Low</span>
                                    {% elif a.overall_risk == 1 %}<span class="badge bg-warning text-dark">Mod</span>
                                    {% else %}<span class="badge bg-danger">High</span>{% endif %}
                                </div>
                                <!-- Double Chevron SVG mapping to the row collapse state -->
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="toggle-icon text-muted ms-2" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M1.646 6.646a.5.5 0 0 1 .708 0L8 12.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                                    <path fill-rule="evenodd" d="M1.646 2.646a.5.5 0 0 1 .708 0L8 8.293l5.646-5.647a.5.5 0 0 1 .708.708l-6 6a.5.5 0 0 1-.708 0l-6-6a.5.5 0 0 1 0-.708z"/>
                                </svg>
                            </td>
                        {% else %}
                            <td class="align-middle text-muted">-</td>
                            <td class="align-middle text-muted">-</td>
                            <td class="align-middle text-muted">-</td>
                            <td class="align-middle"><span class="badge bg-secondary">Pending</span></td>
                        {% endif %}
                    </tr>
                    
                    {% if has_assessment %}
                    <tr class="collapse" id="plot-{{ loop.index }}">
                        <td colspan="6" class="p-0 bg-light text-center border-bottom">
                            <div class="p-3">
                                <a href="/analyses/{{ instance_id }}/kinematic_plot_{{ loop.index0 }}.png" target="_blank">
                                    <img src="/analyses/{{ instance_id }}/kinematic_plot_{{ loop.index0 }}.png" class="img-fluid rounded shadow-sm border" style="width: 100%; object-fit: contain;">
                                </a>
                            </div>
                        </td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</div>
<!-- Prevent screen jumping during mass-collapsing -->
<div style="min-height: 100vh;"></div>
{% endif %}
""" + HTML_BOTTOM
# endregion


# region Routes
@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/analyses')
def list_analyses():
    dirs = [d for d in glob.glob(os.path.join(ANALYSES_DIR, '*')) if os.path.isdir(d)]
    dirs.sort(reverse=True) # Sort descending by time

    instances = []
    for d in dirs:
        instance_id = os.path.basename(d)
        note_path = os.path.join(d, 'note.txt')
        note = open(note_path).read() if os.path.exists(note_path) else "No note"

        has_analysis = os.path.exists(os.path.join(d, 'smashes.json'))
        has_assessment = os.path.exists(os.path.join(d, 'assessments.json'))
        instances.append({
            'id': instance_id,
            'note': note,
            'has_analysis': has_analysis,
            'has_assessment': has_assessment
        })

    return render_template_string(LIST_HTML, instances=instances)

@app.route('/analyses/<instance_id>/delete', methods=['POST'])
def delete_analysis(instance_id):
    dir_path = os.path.join(ANALYSES_DIR, instance_id)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        flash(f"Instance {instance_id} deleted.", "success")
    return redirect(url_for('list_analyses'))

@app.route('/analyses/new', methods=['GET', 'POST'])
def new_analysis():
    if request.method == 'POST':
        video_file = request.files.get('video')
        note = request.form.get('note', '')

        if not video_file or video_file.filename == '':
            flash("No video selected.", "error")
            return redirect(request.url)

        # Generate instance ID and setup directories
        instance_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_dir = os.path.join(ANALYSES_DIR, instance_id)
        os.makedirs(instance_dir)

        with open(os.path.join(instance_dir, 'note.txt'), 'w') as f:
            f.write(note)

        ext = os.path.splitext(video_file.filename)[1]
        input_video_path = os.path.join(instance_dir, f'input{ext}')
        video_file.save(input_video_path)

        # Initialize meta data. Dec_option defaults to auto.
        with open(os.path.join(instance_dir, 'meta.json'), 'w') as f:
            json.dump({
                "input_filename": f'input{ext}',
                "dec_option": "auto",
                "custom_dec": ""
            }, f)

        flash("Video successfully uploaded. Ready for Smash Analysis.", "success")
        return redirect(url_for('detail_analysis', instance_id=instance_id))

    return render_template_string(NEW_HTML)

@app.route('/analyses/<instance_id>/analyze', methods=['POST'])
def analyze_smashes(instance_id):
    instance_dir = os.path.join(ANALYSES_DIR, instance_id)
    if not os.path.exists(instance_dir):
        flash("Analysis not found.", "error")
        return redirect(url_for('list_analyses'))

    meta_path = os.path.join(instance_dir, 'meta.json')
    meta = json.load(open(meta_path))
    input_video_path = os.path.join(instance_dir, meta['input_filename'])

    try:
        # Run Phase 1 Processing (Extraction & Detection)
        metrics, fps = extract_kinematic_metrics(input_video_path)
        smashes = find_smashes(metrics, fps)

        save_json(os.path.join(instance_dir, 'kinematics.json'), metrics)
        save_json(os.path.join(instance_dir, 'smashes.json'), smashes)

        # Update meta with extracted FPS
        meta['fps'] = fps
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        flash("Smash analysis complete. Ready for Risk Assessment.", "success")
    except Exception as e:
        flash(f"Analysis failed: {str(e)}", "error")

    return redirect(url_for('detail_analysis', instance_id=instance_id))

@app.route('/analyses/<instance_id>', methods=['GET', 'POST'])
def detail_analysis(instance_id):
    instance_dir = os.path.join(ANALYSES_DIR, instance_id)
    if not os.path.exists(instance_dir):
        flash("Analysis not found.", "error")
        return redirect(url_for('list_analyses'))

    # Load required data
    meta_path = os.path.join(instance_dir, 'meta.json')
    meta = json.load(open(meta_path))
    fps = meta.get('fps', None)
    input_video_path = os.path.join(instance_dir, meta['input_filename'])

    smashes = load_json(os.path.join(instance_dir, 'smashes.json'), SmashEvent)
    has_analysis = smashes is not None

    # Auto-calculate max deceleration to pre-fill UI (only if analysis is done)
    auto_max_dec = max([abs(s.critical_deceleration) for s in smashes]) if smashes else 0.0

    if request.method == 'POST':
        if not has_analysis:
            flash("Please run Smash Analysis first.", "error")
            return redirect(request.url)

        # Save Assessment Params to restore them next time
        dec_option = request.form.get('dec_option', 'auto')
        custom_dec = request.form.get('custom_dec', '')

        meta['dec_option'] = dec_option
        meta['custom_dec'] = custom_dec
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        # Phase 2: Assessment & Visualization Output Generation
        if dec_option == 'custom' and custom_dec:
            max_dec = float(custom_dec)
        else:
            max_dec = auto_max_dec

        metrics = load_json(os.path.join(instance_dir, 'kinematics.json'), FrameMetrics)

        # 1. Assess Risks
        assessments = assess_smashes(metrics, smashes, max_critical_deceleration=max_dec)
        save_json(os.path.join(instance_dir, 'assessments.json'), assessments)

        # 2. Generate Independent Plots Per Smash
        plot_smashes_kinematics(metrics, fps, smashes, output_dir=instance_dir)

        # 3. Generate Overlay Video
        vid_path = os.path.join(instance_dir, 'analyzed.mp4')
        overlay_kinematic_assessment(
            input_video_path=input_video_path,
            output_video_path=vid_path,
            metrics=metrics,
            smashes=smashes,
            assessments=assessments,
            overwrite=True
        )

        flash("Assessment successfully generated.", "success")
        return redirect(request.url)

    # Render GET View
    assessments = load_json(os.path.join(instance_dir, 'assessments.json'), SmashAssessment)
    return render_template_string(
        DETAIL_HTML,
        instance_id=instance_id,
        input_filename=meta.get('input_filename'),
        fps=fps,
        auto_max_dec=auto_max_dec,
        saved_dec_option=meta.get('dec_option', 'auto'),
        saved_custom_dec=meta.get('custom_dec', ''),
        has_analysis=has_analysis,
        smashes=smashes or [],
        has_assessment=(assessments is not None),
        assessments=assessments or []
    )

# Static file serving for the generated outputs
@app.route('/analyses/<instance_id>/<filename>')
def serve_file(instance_id, filename):
    return send_from_directory(os.path.join(ANALYSES_DIR, instance_id), filename)

# Custom absolute value filter for Jinja templating
@app.template_filter('abs')
def absolute_value(number):
    return abs(number)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
# endregion