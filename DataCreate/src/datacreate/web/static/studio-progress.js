// Shared by the full Studio and the dedicated upload-only page.
export function createProgressView() {
const analysisPanel = document.createElement('section');
analysisPanel.id = 'analysis-progress'; analysisPanel.className = 'analysis-progress'; analysisPanel.hidden = true;
analysisPanel.setAttribute('aria-labelledby', 'analysis-title');
analysisPanel.innerHTML = '<div class="analysis-progress-heading"><p class="eyebrow">YOUR PERFORMANCE → YOUR FEEDBACK</p><h2 id="analysis-title">Bringing your feedback to life</h2><p id="analysis-current" role="status" aria-live="polite"></p></div><div class="analysis-meter"><span id="analysis-count"></span><progress id="analysis-bar" max="16" value="0" aria-label="Completed pipeline milestones"></progress><p id="analysis-units" hidden></p></div><ol id="analysis-steps"></ol><p class="analysis-note">Progress follows completed tasks, not elapsed time. Some tasks take longer than others.</p>';

const $ = id => analysisPanel.querySelector(`#${id}`);
let progressViewKey = '';
function waiting(message) {
  analysisPanel.hidden = false; analysisPanel.dataset.status = 'processing';
  $('analysis-title').textContent = 'Bringing your feedback to life';
  $('analysis-current').textContent = message;
  $('analysis-count').textContent = 'Preparing your take';
  $('analysis-bar').removeAttribute('value');
  $('analysis-units').hidden = true; $('analysis-steps').replaceChildren();
  progressViewKey = '';
}
function render(progress, legacy) {
  // Older servers can still display their original stage milestones.
  if (!progress && legacy) progress = {...legacy, phases: legacy.steps.map(s => ({...s, substeps: []}))};
  analysisPanel.hidden = !progress;
  if (!progress) return;
  const key = JSON.stringify(progress);
  if (key === progressViewKey) return;
  progressViewKey = key;
  analysisPanel.dataset.status = progress.status || 'processing';
  $('analysis-title').textContent = progress.status === 'complete' ? 'Your feedback is ready' : progress.status === 'failed' ? 'This take needs attention' : 'Bringing your feedback to life';
  $('analysis-current').textContent = progress.message;
  $('analysis-count').textContent = `${progress.completed} / ${progress.total} milestones resolved`;
  $('analysis-bar').max = progress.total; $('analysis-bar').value = progress.completed;
  $('analysis-bar').setAttribute('aria-valuetext', `${progress.completed} of ${progress.total} milestones completed or skipped. ${progress.message}`);
  $('analysis-units').hidden = !progress.units;
  if (progress.units) $('analysis-units').textContent = `${progress.units.completed.toLocaleString()} of ${progress.units.total.toLocaleString()} completed in this task`;
  const labels = {complete: 'Done', skipped: 'Skipped / reused', active: 'In progress', pending: 'Waiting', failed: 'Stopped'};
  $('analysis-steps').replaceChildren(...progress.phases.map((step, index) => {
    const item = document.createElement('li'); item.className = step.state;
    const heading = document.createElement('div'); heading.className = 'phase-heading';
    const number = document.createElement('span'); number.className = 'phase-number'; number.textContent = step.state === 'complete' ? '✓' : String(index+1).padStart(2, '0');
    const label = document.createElement('strong'); label.textContent = step.label;
    const status = document.createElement('span'); status.className = 'phase-status'; status.textContent = labels[step.state];
    heading.append(number, label, status);
    if (step.state === 'active') item.setAttribute('aria-current', 'step');
    const children = document.createElement('ul');
    for (const substep of step.substeps) {
      const row = document.createElement('li'); row.className = substep.state;
      const marker = document.createElement('span'); marker.className = 'task-marker'; marker.setAttribute('aria-hidden', 'true');
      marker.textContent = {complete: '✓', skipped: '–', active: '●', pending: '○', failed: '!'}[substep.state];
      const text = document.createElement('span'); text.textContent = substep.label;
      row.setAttribute('aria-label', `${substep.label}: ${labels[substep.state]}`);
      row.append(marker, text); children.append(row);
    }
    item.append(heading, children); return item;
  }));
}
return {element: analysisPanel, waiting, render};
}
