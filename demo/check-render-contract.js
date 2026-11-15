// Offline rendering contract only: no browser, network, inference or new judgments.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
const folder = __dirname;
const html = fs.readFileSync(path.join(folder, 'index.html'), 'utf8');
assert.equal(fs.readFileSync(path.join(folder, 'promptfix-confirmation.html'), 'utf8'), html);
const data = JSON.parse(fs.readFileSync(path.join(folder, 'promptfix-confirmation.json'), 'utf8'));
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.ok(html.includes('data-brand-theme="dark"'));
assert.ok(html.includes('class="brand-ui"'));
assert.ok(html.includes('brand/255labs-lockup.svg'));
assert.ok(html.indexOf('id="quality-criterion"') < html.indexOf('id="question"'));
assert.ok(html.includes('Vary target system prompts'));
for (const name of ['brand/255labs-tokens.css', 'brand/255labs-ui.css', 'explorer.css']) {
  assert.ok(html.includes('href="' + name + '"'));
  assert.ok(fs.statSync(path.join(folder, name)).size > 0);
}
const crypto = require('crypto');
const brand = JSON.parse(fs.readFileSync(path.join(folder, 'brand/255labs-manifest.json'), 'utf8'));
assert.equal(brand.version, '1.1.0');
for (const [name, expected] of Object.entries(brand.files)) {
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(folder, 'brand', name))).digest('hex'), expected);
}
for (const prefix of ['off', 'on', 'prompt']) {
  assert.ok(html.includes('<blockquote class="response" id="' + prefix + '-text">'));
  assert.ok(html.indexOf(`id="${prefix}-score"`) < html.indexOf(`id="${prefix}-text"`), 'Score must precede answer');
  assert.ok(html.indexOf(`id="${prefix}-verdict"`) < html.indexOf(`id="${prefix}-text"`), 'Verdict must precede answer');
}

class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.value = '0';
    this.textContent = '';
    this.children = [];
    this.events = {};
    this.attributes = {};
    this.classList = {add() {}, remove() {}};
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  addEventListener(type, handler) { this.events[type] = handler; }
  setAttribute(name, value) { this.attributes[name] = value; }
}

async function run(query) {
  const nodes = Object.fromEntries([...html.matchAll(/\bid="([^"]+)"/g)].map(m => [m[1], new Element()]));
  const context = {
    document: {
      getElementById(id) { assert.ok(nodes[id], 'Unknown HTML id: ' + id); return nodes[id]; },
      createElement(tag) { return new Element(tag); },
    },
    location: {href: 'https://example.test/promptfix-confirmation.html' + query},
    URL,
    fetch: async url => {
      assert.equal(url, 'promptfix-confirmation.json');
      return {ok: true, json: async () => data};
    },
  };
  vm.createContext(context);
  assert.equal((script.match(/start\(\)\.catch/g) || []).length, 1);
  vm.runInContext(script.replace('start().catch', 'globalThis.ready=start().catch'), context);
  await context.ready;
  assert.notEqual(nodes.loading.className, 'error', nodes.loading.textContent);
  assert.equal(nodes.question.children.length, 64);
  assert.equal(nodes.model.children.length, 2);
  return nodes;
}

(async () => {
  const directId = 'behavior-promptfix-fresh64-031';
  const nodes = await run('?case=' + directId);
  assert.equal(nodes['question-text'].textContent, data.cases.find(c => c.id === directId).question);
  const covered = new Set();
  const badges = {won: 0, not_better: 0, tied: 0};
  let rendered = 0;
  for (let mi = 0; mi < data.models.length; mi++) {
    nodes.model.value = String(mi);
    nodes.model.events.change();
    const model = data.models[mi];
    const strengths = [0, ...model.metrics.map(m => m.strength).sort((a, b) => a - b)];
    assert.equal(Number(nodes.strength.max), strengths.length - 1);
    for (let qi = 0; qi < data.cases.length; qi++) {
      const q = data.cases[qi];
      nodes.question.value = String(qi);
      nodes.question.events.change();
      for (let si = 0; si < strengths.length; si++) {
        nodes.strength.value = String(si);
        nodes.strength.events.input();
        const strength = strengths[si];
        const record = strength === 0 ? q.off : q.models[model.id][String(strength)];
        assert.equal(nodes['question-text'].textContent, q.question);
        assert.equal(nodes['off-text'].textContent, q.off.output);
        assert.equal(nodes['prompt-text'].textContent, q.prompt_only.output);
        assert.equal(nodes['on-text'].textContent, record.output);
        assert.equal(nodes['on-score'].textContent, Number(record.effect).toFixed(4) + ' / 1');
        for (const [prefix, answer] of [['on', record], ['prompt', q.prompt_only]]) {
          const gain = Number(answer.effect) - Number(q.off.effect);
          const won = gain > 0;
          assert.equal(nodes[prefix + '-verdict'].className, 'verdict ' + (won ? 'won' : 'not-better'));
          assert.ok(nodes[prefix + '-verdict'].textContent.startsWith(won ? '✅ Beats original' : '❌ Does not beat original'));
          badges[won ? 'won' : 'not_better']++;
          if (gain === 0) badges.tied++;
        }
        assert.ok(nodes['on-termination'].textContent.includes('end-of-response token'));
        assert.equal(nodes.metrics.children.length, 2 + model.metrics.length);
        covered.add(q.id + '|off');
        covered.add(q.id + '|prompt-only');
        if (strength !== 0) covered.add(q.id + '|' + model.metrics.find(m => m.strength === strength).source_arm);
        rendered++;
      }
    }
  }
  assert.equal(covered.size, 384);
  assert.ok(badges.won > 0 && badges.not_better > 0 && badges.tied > 0);
  const fallback = await run('?case=unknown-question');
  assert.equal(fallback['question-text'].textContent, data.cases[0].question);
  const result = {
    state: 'pass-offline-recorded-answer-rendering-contract',
    rendered_states: rendered,
    exact_saved_responses_covered: covered.size,
    direct_question_link_verified: directId,
    invalid_question_link_falls_back: true,
    variable_strength_count_per_model_verified: true,
    full_answer_text_matches_export_without_truncation: true,
    scores_and_verdicts_precede_answers: true,
    green_only_for_strict_improvement_and_red_including_ties: badges,
    homepage_restored_full_example_selector: data.cases.length,
    homepage_all_checkpoints_and_strengths_restored: true,
    complete_responses_are_blockquotes: true,
    criterion_precedes_examples: true,
    dark_theme_and_pinned_brand_kit_verified: brand.version,
    exact_original_brand_files_verified: Object.keys(brand.files).length,
    browser_visual_or_click_QA: 'unverified; this is an offline DOM contract, not a browser',
    network_calls: 0,
    new_model_or_Jev_calls: 0,
  };
  const output = path.join(folder, 'render-contract-restored-dark-explorer-v4.json');
  assert.ok(!fs.existsSync(output));
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  process.stdout.write(JSON.stringify(result) + '\n');
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
