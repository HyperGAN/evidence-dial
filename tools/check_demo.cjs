// DOM behavior check of our static explorer. No browser or model inference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { JSDOM } = require('../.ui-test/node_modules/jsdom');
const arms = (suffix) => ({off:{output:'OFF '+suffix,success:0},particle_1:{output:'ONE '+suffix,success:1},particle_2:{output:'TWO '+suffix,success:0},lora:{output:'LORA '+suffix,success:1}});
const fixture = {status:'Test fixture',strengths:[1,2],stats:[],rates:[],scope:'Test only',links:[],cases:[
  {id:'a',dataset:'policy',case_type:'answerable',question:'Question A',context:'Context A',gold:'A',arms:arms('A')},
  {id:'b',dataset:'policy',case_type:'missing',question:'Question B',context:'Context B',gold:'UNKNOWN',arms:arms('<img src=x onerror=alert(1)>')},
  {id:'c',dataset:'faith',case_type:'conflict',question:'Question C',context:'Original source link',gold:'CONFLICT',arms:arms('C')}
]};
(async () => {
  const dom = new JSDOM(fs.readFileSync('demo/index.html','utf8'), {runScripts:'dangerously',url:'https://example.test/',beforeParse(w){w.fetch=async()=>({ok:true,json:async()=>fixture});}});
  await new Promise(resolve=>setImmediate(resolve));
  const d=dom.window.document, byId=id=>d.getElementById(id), change=id=>byId(id).dispatchEvent(new dom.window.Event('change'));
  assert.equal(d.querySelectorAll('.answer').length,3);
  assert.match(byId('answers').textContent,/ONE A/);
  d.querySelector('[data-strength="2"]').click();
  assert.match(byId('answers').textContent,/TWO A/);
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
  console.log(JSON.stringify({passed:true,checks:['three matched arms','strength switch','previous/next wrap','case-type filter','empty-filter clearing','dataset switch','model text escaped'],runtime:process.version,visual_browser_check:false}));
  dom.window.close();
})().catch(e=>{console.error(e);process.exitCode=1});
