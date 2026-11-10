// DOM behavior check of our static explorer. No browser or model inference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { JSDOM } = require('../.ui-test/node_modules/jsdom');
const arms = (suffix) => ({off:{output:'OFF '+suffix,success:0},particle_1:{output:'ONE '+suffix,success:1},particle_2:{output:'TWO '+suffix,success:0},lora:{output:'LORA '+suffix,success:1},hinted_particle_1:{output:'HINTONE '+suffix,success:0},hinted_particle_2:{output:'HINTTWO '+suffix,success:1},lora_original:{output:'OLDLORA '+suffix,success:1}});
const fixture = {status:'Test fixture',strengths:[1,2],experiments:{curated:{label:'Curated',particle_prefix:'particle_',lora_key:'lora'},hinted:{label:'Hinted',particle_prefix:'hinted_particle_',lora_key:'lora_original'}},dataset_strengths:{policy:[1,2],faith:[1,2],balanced:[1]},stats:[],rates:[],scope:'Test only',links:[],cases:[
  {id:'a',dataset:'policy',case_type:'answerable',question:'Question A',context:'Context A',gold:'A',arms:arms('A')},
  {id:'b',dataset:'policy',case_type:'missing',question:'Question B',context:'Context B',gold:'UNKNOWN',arms:arms('<img src=x onerror=alert(1)>')},
  {id:'c',dataset:'faith',case_type:'conflict',question:'Question C',context:'Original source link',gold:'CONFLICT',arms:arms('C')},
  {id:'d',dataset:'balanced',case_type:'answerable',question:'Question D',context:'Fresh test',gold:'D',arms:Object.fromEntries(Object.entries(arms('D')).filter(([a])=>!['particle_2','hinted_particle_2'].includes(a)))}
]};
(async () => {
  const dom = new JSDOM(fs.readFileSync('demo/index.html','utf8'), {runScripts:'dangerously',url:'https://example.test/',beforeParse(w){w.fetch=async()=>({ok:true,json:async()=>fixture});}});
  await new Promise(resolve=>setImmediate(resolve));
  const d=dom.window.document, byId=id=>d.getElementById(id), change=id=>byId(id).dispatchEvent(new dom.window.Event('change'));
  assert.equal(d.querySelectorAll('.answer').length,3);
  assert.match(byId('answers').textContent,/ONE A/);
  d.querySelector('[data-strength="2"]').click();
  assert.match(byId('answers').textContent,/TWO A/);
  assert.equal(d.querySelector('[data-strength="2"]').getAttribute('aria-pressed'),'true');
  byId('prev').click();
  assert.equal(byId('question').textContent,'Question B');
  assert.equal(byId('answers').querySelectorAll('img').length,0);
  assert.match(byId('answers').textContent,/<img src=x/);
  byId('next').click();
  assert.equal(byId('question').textContent,'Question A');
  byId('type').value='missing';change('type');
  assert.equal(byId('question').textContent,'Question B');
  byId('type').value='altered';change('type');
  assert.equal(byId('answers').children.length,0);
  assert.equal(byId('context').textContent,'');
  assert.equal(byId('gold').textContent,'');
  assert.match(byId('counter').textContent,/No cases/);
  byId('type').value='all';change('type');
  byId('dataset').value='faith';change('dataset');
  assert.equal(byId('question').textContent,'Question C');
  assert.equal(d.querySelectorAll('.answer').length,3);
  byId('dataset').value='balanced';change('dataset');
  assert.equal(d.querySelector('[data-strength="2"]').disabled,true);
  assert.equal(d.querySelector('[data-strength="1"]').getAttribute('aria-pressed'),'true');
  assert.match(byId('answers').textContent,/ONE D/);
  assert.doesNotMatch(byId('answers').textContent,/UNMEASURED/);
  byId('dataset').value='policy';change('dataset');
  assert.equal(d.querySelector('[data-strength="2"]').disabled,false);
  byId('experiment').value='hinted';change('experiment');
  assert.match(byId('answers').textContent,/HINTONE A/);
  assert.match(byId('answers').textContent,/OLDLORA A/);
  byId('experiment').value='curated';change('experiment');
  assert.match(byId('answers').textContent,/ONE A/);
  assert.doesNotMatch(byId('answers').textContent,/OLDLORA/);
  console.log(JSON.stringify({passed:true,checks:['three matched arms','strength switch','previous/next wrap','case-type filter','empty-filter clearing','dataset switch','model text escaped'],runtime:process.version,visual_browser_check:false}));
  dom.window.close();
})().catch(e=>{console.error(e);process.exitCode=1});
