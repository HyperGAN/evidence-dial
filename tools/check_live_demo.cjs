// Checks the actual public HTTP assets in JSDOM. This is not a browser visual test.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM}=require('../.ui-test/node_modules/jsdom');
(async()=>{
 const url=process.argv[2],out=process.argv[3];assert(url&&out,'Supply actual static host URL and a new output receipt');
 assert(!fs.existsSync(out),'Preserve existing checks; use a new receipt');
 const [htmlResponse,dataResponse]=await Promise.all([fetch(url+'/index.html'),fetch(url+'/benchmark.json')]);
 assert(htmlResponse.ok&&dataResponse.ok);
 const html=await htmlResponse.text(),data=await dataResponse.json();
 const dom=new JSDOM(html,{runScripts:'dangerously',url:url+'/',beforeParse(w){w.fetch=async path=>{assert.equal(path,'benchmark.json');return {ok:true,json:async()=>data}}}});
 await new Promise(r=>setImmediate(r));
 const d=dom.window.document,byId=id=>d.getElementById(id),change=id=>byId(id).dispatchEvent(new dom.window.Event('change'));
 assert.equal(d.querySelectorAll('#results-body tr').length,data.aggregates.length);
 const pc=x=>x==null?'unmeasured':(x*100).toFixed(1)+'%';
 for(let i=0;i<data.aggregates.length;i++){
  const row=data.aggregates[i],cells=[...d.querySelectorAll('#results-body tr')[i].children].map(c=>c.textContent);
  assert.deepEqual(cells.slice(0,6),[row.dataset,String(row.cases),...['off','particle_1','particle_2','lora'].map(k=>pc(row.success_rates[k]))]);
 }
 const views=[];
 for(const [variant,config] of Object.entries(data.experiments)){
  byId('experiment').value=variant;change('experiment');
  for(const [dataset,strengths] of Object.entries(data.experiment_strengths[variant])){
   byId('dataset').value=dataset;change('dataset');
   const rows=data.cases.filter(c=>c.dataset===dataset);
   for(const strength of strengths){
    byId('type').value='all';change('type');
    const button=d.querySelector(`[data-strength="${strength}"]`);assert(!button.disabled);button.click();
    assert.equal(button.getAttribute('aria-pressed'),'true');
    assert.equal(byId('counter').textContent,`1 / ${rows.length} · frozen sample order`);
    let filters=0;
    for(const type of new Set(rows.map(c=>c.case_type))){
     byId('type').value=type;change('type');
     const c=rows.find(c=>c.case_type===type),keys=['off',config.particle_prefix+strength,config.lora_key];
     const cards=[...d.querySelectorAll('.answer')];assert.equal(cards.length,3);
     for(let j=0;j<keys.length;j++){
      assert.equal(cards[j].querySelector('.output').textContent,c.arms[keys[j]].output);
      assert.equal(cards[j].querySelector('.pill').textContent,c.arms[keys[j]].success?'PASS':'MISS');
     }
     filters++;
    }
    byId('type').value='all';change('type');
    views.push({variant,dataset,strength,cases:rows.length,case_type_filters_verified:filters});
   }
  }
 }
 assert.equal(data.cases.length,2772);assert.equal(data.aggregates.length,10);assert.equal(views.length,18);
 const statText=byId('stats').textContent;assert(statText.includes('-34.9 pp'));assert(statText.includes('+25.7 pp'));assert(statText.includes('17,868'));
 const result={passed:true,url,cases:data.cases.length,aggregate_rows:data.aggregates.length,views,
  runtime:process.version,checks:['actual public HTTP assets','all measured variant/dataset/strength views','case-type filters','matched native answer text and strict verdicts','aggregate rates','positive and negative summary stats'],
  real_browser_visual_QA:false,model_or_API_inference_calls:0};
 fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify({passed:true,cases:result.cases,aggregate_rows:result.aggregate_rows,views:views.length,real_browser_visual_QA:false}));dom.window.close();
})().catch(e=>{console.error(e);process.exitCode=1});
