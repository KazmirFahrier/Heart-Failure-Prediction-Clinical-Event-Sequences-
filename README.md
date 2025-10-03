<h1>Heart Failure Prediction — End-to-End ML Pipeline</h1>

<p>
This project builds a complete workflow to predict Heart Failure (HF) from longitudinal clinical events:
data prep &amp; cohort stats → feature engineering → SVMLight export → model training &amp; validation
(Logistic Regression, Linear SVM, Decision Tree) with K-Fold and Randomized cross-validation.
</p>

<hr/>

<h2>📁 Project Layout</h2>
<pre><code>.
├─ data/
│  ├─ train/
│  │  ├─ events.csv
│  │  ├─ hf_events.csv
│  │  └─ event_feature_map.csv
│  └─ val/
│     ├─ events.csv
│     ├─ hf_events.csv
│     └─ event_feature_map.csv
├─ features_svmlight.train
├─ features_svmlight.val
├─ HW1_pre.ipynb
├─ utils.py
└─ README.md
</code></pre>

<hr/>

<h2>🧠 Problem &amp; Data</h2>
<ul>
  <li><strong>Goal:</strong> Predict whether a patient will develop HF from their sequence of clinical events.</li>
  <li><strong>events.csv</strong>: tuples <code>(pid, event_id, vid, value)</code> — ordinal visit index <code>vid</code>, value is 1 in this synthesized set.</li>
  <li><strong>hf_events.csv</strong>: tuples <code>(pid, vid, label=1)</code> for patients with HF; <code>vid</code> is the first HF visit; patients not listed are non-HF.</li>
  <li><strong>event_feature_map.csv</strong>: maps <code>event_id → feature_id (idx)</code>.</li>
</ul>

<hr/>

<h2>🛠️ Environment</h2>
<pre><code>python &gt;= 3.9
pip install -r requirements.txt
</code></pre>

<p><strong>requirements.txt</strong></p>
<pre><code>numpy
pandas
scikit-learn
scipy
</code></pre>

<hr/>

<h2>🚦 Pipeline</h2>

<h3>1) Descriptive statistics</h3>
<p>Compute two metrics for HF vs non-HF cohorts:</p>
<ul>
  <li><strong>Event count</strong> per patient (number of rows in <code>events.csv</code>).</li>
  <li><strong>Encounter count</strong> per patient (unique visit count per <code>pid</code>).</li>
</ul>

<p><strong>Reference outputs (train split)</strong></p>
<ul>
  <li>Event counts (avg, max, min): HF = <code>188.9375, 2046, 28</code>; non-HF = <code>118.64423076923077, 1014, 6</code>.</li>
  <li>Encounter counts (avg, max, min): HF = <code>2.8060810810810812, 34, 2</code>; non-HF = <code>2.189423076923077, 11, 1</code>.</li>
</ul>

<h3>2) Feature construction</h3>
<ol>
  <li><strong>Index visit:</strong>
    <ul>
      <li>HF patients → <em>first</em> HF visit (from <code>hf_events.csv</code>).</li>
      <li>Non-HF patients → <em>last</em> visit (max <code>vid</code> in <code>events.csv</code>).</li>
    </ul>
    Output shape: <code>(4000, 2)</code> with columns <code>pid, indx_vid</code>.
  </li>
  <li><strong>Filter to observation window:</strong> keep events with <code>vid &lt; indx_vid</code>; return <code>pid, event_id, value</code>.
  </li>
  <li><strong>Aggregate &amp; normalize:</strong>
    <ul>
      <li>Count occurrences per <code>(pid, event_id)</code>.</li>
      <li>Map <code>event_id → feature_id</code>.</li>
      <li>Min-max (per feature): <code>value := value / max(value)</code> (min assumed 0).</li>
    </ul>
  </li>
  <li><strong>Save SVMLight:</strong> for each patient, write
    <code>&lt;label&gt; &lt;feature_id&gt;:&lt;value&gt; ...</code> with patients sorted by <code>pid</code> and features sorted by <code>feature_id</code>.
    Skip zeros. Files: <code>features_svmlight.train</code>, <code>features_svmlight.val</code>.
  </li>
</ol>

<h3>3) Modeling</h3>
<p>Train on <code>features_svmlight.train</code>; evaluate on both train (3.1a) and val (3.1b).</p>
<ul>
  <li>Logistic Regression (defaults)</li>
  <li>Linear SVM (<code>random_state=545510477</code>)</li>
  <li>Decision Tree (<code>max_depth=5</code>, <code>random_state=545510477</code>)</li>
</ul>

<p><strong>Example results (val split)</strong></p>
<table>
  <thead>
    <tr><th>Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th></tr>
  </thead>
  <tbody>
    <tr><td>Logistic Regression</td><td>0.6937086093</td><td>0.7345360825</td><td>0.7765667575</td><td>0.7549668874</td></tr>
    <tr><td>Linear SVM</td><td>0.6407284768</td><td>0.7038043478</td><td>0.7057220708</td><td>0.7047619048</td></tr>
    <tr><td>Decision Tree (d=5)</td><td>0.6821192053</td><td>0.6611418048</td><td>0.9782016349</td><td>0.7890109890</td></tr>
  </tbody>
</table>

<h3>4) Cross-Validation</h3>
<ul>
  <li><strong>K-Fold (k=5)</strong>: mean F1 ≈ <code>0.7171</code> (example run)</li>
  <li><strong>Randomized CV</strong> (ShuffleSplit, <code>test_size=0.2</code>, <code>n_splits=5</code>): mean F1 ≈ <code>0.7196</code></li>
</ul>

<hr/>

<h2>🧪 Reproducibility</h2>
<ul>
  <li>Use seed <code>545510477</code> for Linear SVM and Decision Tree.</li>
  <li>Normalize <em>per feature</em>, not per patient.</li>
  <li>Ensure deterministic SVMLight export (sorted patients &amp; features; omit zeros).</li>
</ul>

<hr/>

<h2>⚠️ Challenges &amp; How They Were Solved</h2>
<ol>
  <li>
    <strong>Hidden tests failed for SVM (precision/F1 mismatch)</strong><br/>
    <em>Cause:</em> Model functions referenced a module-level <code>RANDOM_STATE</code> that isn’t present when the grader imports the functions in isolation.<br/>
    <em>Fix:</em> Made each function self-contained: import inside the function and define the seed locally (e.g., <code>seed = 545510477</code>) so results are stable and match the grader.
  </li>
  <li>
    <strong>NameError for imports (e.g., <code>LogisticRegression</code>)</strong><br/>
    <em>Cause:</em> Running tests in a fresh kernel without the imports cell executed.<br/>
    <em>Fix:</em> Added local (in-function) imports so evaluation order doesn’t matter.
  </li>
  <li>
    <strong>“Output missing / timeout” for 3.2 cross-validation</strong><br/>
    <em>Cause:</em> Stubbed functions or functions not returning a float; or relying on globals not visible to the grader.<br/>
    <em>Fix:</em> Implemented <code>get_f1_kfold</code> with <code>KFold(shuffle=True, random_state=545510477)</code> and
    <code>get_f1_randomisedCV</code> with <code>ShuffleSplit</code>; both return a plain Python <code>float</code> mean F1.
  </li>
  <li>
    <strong>Potential metric drift from feature prep</strong><br/>
    <em>Cause:</em> Normalizing per patient or unsorted feature export can silently change downstream metrics.<br/>
    <em>Fix:</em> Normalize per <em>feature_id</em> (divide by that feature’s max) and sort
    features by <em>feature_id</em>, patients by <em>pid</em> when writing SVMLight.
  </li>
  <li>
    <strong>dtype pitfalls on <code>vid</code></strong><br/>
    <em>Cause:</em> If <code>vid</code> is parsed as string, comparisons like <code>vid &lt; indx_vid</code> may misbehave.<br/>
    <em>Fix:</em> Ensure integer dtype when computing <code>indx_vid</code> and filtering the observation window.
  </li>
</ol>

<hr/>

<h2>🔌 Quick usage</h2>
<pre><code>from sklearn.datasets import load_svmlight_file
Xtr, ytr = load_svmlight_file("features_svmlight.train")
Xva, yva = load_svmlight_file("features_svmlight.val")
</code></pre>

<hr/>

<h2>📄 License</h2>
<p>MIT.</p>

<hr/>

<h2>🙌 Acknowledgements</h2>
<p>Course staff for dataset preparation, skeleton code, and utilities.</p>
