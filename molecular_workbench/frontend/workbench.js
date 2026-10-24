(() => {
  'use strict';
  const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
  let points=[],pins=[],selected=null,context=null,dataset='',total=0,queryMode=false,view='landscape',lastSent=null,sequence=0;
  const send=(type,extra={})=>parent.postMessage({isStreamlitMessage:true,type,...extra},'*');
  const resize=()=>requestAnimationFrame(()=>send('streamlit:setFrameHeight',{height:Math.ceil(document.body.getBoundingClientRect().height)+4}));
  const element=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const find=id=>points.find(p=>p.id===id);
  function emit(){lastSent=`${Date.now()}-${++sequence}`;send('streamlit:setComponentValue',{value:{context,pins:[...pins],event_id:lastSent},dataType:'json'});}
  function svg(tag,attributes,text){const n=document.createElementNS(ns,tag);Object.entries(attributes).forEach(([k,v])=>n.setAttribute(k,String(v)));if(text!==undefined)n.textContent=text;return n;}
  function extent(values){let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=1;hi+=1;}const pad=(hi-lo)*.08;return[lo-pad,hi+pad];}
  function measurementY(p){return p.value===null?null:(p.measurement_type==='Ki'?(p.value>0?Math.log10(p.value):null):p.value);}
  function draw(){
    $('plot').replaceChildren();$('records').replaceChildren();
    $('plot-wrap').hidden=view==='records';$('records').hidden=view!=='records';
    const endpoint=$('endpoint').value;
    const data=points.map(p=>({p,x:queryMode?p.similarity:p.weight,y:view==='measurements'?measurementY(p):(queryMode?p.weight:p.rings)})).filter(d=>Number.isFinite(d.x)&&Number.isFinite(d.y));
    $('chart-title').textContent=view==='records'?'Source records':view==='measurements'?'Reported measurements':queryMode?'Similarity landscape':'Structure overview';
    $('chart-count').textContent=`${points.length} of ${total} source records`;
    if(view==='records'){
      for(const p of points){const b=element('button','record-row');b.setAttribute('aria-pressed',String(p.id===selected));b.append(element('strong','',p.name),element('span','',queryMode?`Tanimoto ${p.similarity.toFixed(2)}`:`${p.weight.toFixed(1)} g/mol`));b.onclick=()=>inspect(p.id);$('records').append(b);}
    }else if(data.length){
      const [x0,x1]=queryMode?[0,1.03]:extent(data.map(d=>d.x));const [y0,y1]=extent(data.map(d=>d.y));
      const x=v=>70+(v-x0)/(x1-x0)*690,y=v=>345-(v-y0)/(y1-y0)*300;
      for(let i=0;i<=4;i++){
        const xv=x0+(x1-x0)*i/4,yv=y0+(y1-y0)*i/4;
        $('plot').append(svg('line',{x1:70,y1:y(yv),x2:760,y2:y(yv),class:'grid'}));
        $('plot').append(svg('text',{x:58,y:y(yv)+4,'text-anchor':'end'},yv.toFixed(view==='measurements'?1:0)));
        $('plot').append(svg('text',{x:x(xv),y:367,'text-anchor':'middle'},xv.toFixed(queryMode?2:0)));
      }
      $('plot').append(svg('line',{x1:70,y1:345,x2:760,y2:345,class:'axis'}));
      $('plot').append(svg('text',{x:415,y:394,'text-anchor':'middle'},queryMode?'Structural similarity · Tanimoto':'Calculated molecular weight · g/mol'));
      const yLabel=view==='measurements'?(data[0].p.measurement_type==='Ki'?'Ki · log10(nM)':`Measured solubility · ${data[0].p.units}`):(queryMode?'Molecular weight · g/mol':'Calculated ring count');
      $('plot').append(svg('text',{x:16,y:195,transform:'rotate(-90 16 195)','text-anchor':'middle'},yLabel));
      for(const d of data){
        const label=d.p.labels?.[endpoint];let cls='data-point';
        if(d.p.id===selected)cls+=' selected';if(d.p.match==='Exact identity')cls+=' exact';
        if(Object.keys(d.p.labels).length)cls+=' '+(label===1?'active':label===0?'inactive':'missing');
        const dot=svg('circle',{cx:x(d.x),cy:y(d.y),r:d.p.id===selected?7:5.5,class:cls,tabindex:0,role:'button','aria-label':`Inspect ${d.p.name}, source ${d.p.id}`});
        dot.onclick=()=>inspect(d.p.id);dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspect(d.p.id);}};
        dot.onmouseenter=e=>{const r=$('plot-wrap').getBoundingClientRect();$('tooltip').textContent=`${d.p.name} · ${d.p.measurement}`;$('tooltip').style.left=Math.max(0,Math.min(r.width-230,e.clientX-r.left+10))+'px';$('tooltip').style.top=Math.max(0,e.clientY-r.top-45)+'px';$('tooltip').hidden=false;};
        dot.onmouseleave=()=>{$('tooltip').hidden=true;};$('plot').append(dot);
      }
    }else $('plot').append(svg('text',{x:400,y:195,'text-anchor':'middle'},'No plottable measurements in this selection.'));
    $('plot-note').textContent=Object.keys(points[0]?.labels||{}).length?`${endpoint}: orange = active, blue = inactive, gray = not reported. Screening labels do not establish clinical toxicity or safety.`:view==='measurements'?'Values retain their dataset units. Assay conditions may differ; proximity in this chart does not establish comparable experiments.':'Each point is one source record. Descriptors are calculated with RDKit; chart position does not establish biological activity.';
    document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===view)));
  }
  function inspect(id){selected=id;draw();renderInspector();}
  function renderInspector(){
    const p=find(selected);if(!p)return;
    $('selected-name').textContent=p.name;$('selected-match').textContent=p.match+(queryMode?` · Tanimoto ${p.similarity.toFixed(2)}`:'');
    $('structure').src=p.image;$('structure').alt=`Molecular structure of ${p.name}`;$('measurement').textContent=p.measurement;
    $('properties').replaceChildren();
    for(const [name,value] of [['Mol. weight (g/mol)',p.weight],['Calculated logP',p.logp],['H-bond donors',p.donors],['H-bond acceptors',p.acceptors]]){const d=element('div');d.append(element('dt','',name),element('dd','',String(value)));$('properties').append(d);}
    $('pin').textContent=pins.includes(p.id)?'Unpin source':'Pin to comparison';$('source').href=p.url;$('source-text').textContent=p.text;$('source-id').textContent=`Source ID: ${p.id}`;
    $('endpoint-labels').replaceChildren();for(const [key,value] of Object.entries(p.labels)){const row=element('div','',`${key}: ${value===null?'Not reported':value===1?'1 (active)':'0 (inactive)'}`);$('endpoint-labels').append(row);}
    resize();
  }
  function renderTray(){
    $('tray').replaceChildren();$('pin-count').textContent=`${pins.length} / 3`;$('export').disabled=!pins.length;
    for(let i=0;i<3;i++){
      const p=find(pins[i]);const card=element('article','tray-card'+(p?'':' empty'));
      if(p){const image=element('img');image.src=p.image;image.alt=`Structure of ${p.name}`;const remove=element('button','remove','×');remove.setAttribute('aria-label',`Unpin ${p.name}`);remove.onclick=()=>togglePin(p.id);card.append(remove,element('strong','',p.name),image,element('p','',p.measurement),element('p','',`MW ${p.weight} g/mol · logP ${p.logp}`));}
      else card.append(element('span','',`COMPARISON ${i+1}`),element('span','','Select a point, then pin its source.'));
      $('tray').append(card);
    }
    resize();
  }
  function togglePin(id){
    if(pins.includes(id))pins=pins.filter(i=>i!==id);else if(pins.length<3)pins.push(id);else{$('feedback').textContent='Three sources are pinned. Unpin one to add another.';return;}
    $('feedback').textContent=`${pins.length} source records pinned. Pinned sources will be used for the optional summary.`;renderTray();renderInspector();emit();
  }
  $('pin').onclick=()=>{if(selected)togglePin(selected);};
  $('copy').onclick=async()=>{const p=find(selected);if(!p)return;try{await navigator.clipboard.writeText(p.smiles);$('feedback').textContent='SMILES copied.';}catch{$('feedback').textContent='Copy this SMILES: '+p.smiles;}resize();};
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;draw();resize();});
  $('endpoint').onchange=()=>{draw();resize();};
  $('export').onclick=()=>{const data={dataset,source_records:pins.map(find).map(({image,...p})=>p)};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=element('a');a.href=url;a.download='molescope-comparison.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);};
  window.addEventListener('message',event=>{
    if(event.source!==parent||event.data.type!=='streamlit:render')return;
    const a=event.data.args;
    if(context===a.context&&lastSent&&a.ack!==lastSent)return;
    if(context!==a.context){context=a.context;lastSent=null;selected=null;view='landscape';$('feedback').textContent='';}
    points=a.points||[];pins=(a.pinned||[]).filter(id=>points.some(p=>p.id===id));dataset=a.dataset;total=a.total;queryMode=a.query_mode;
    if(!find(selected))selected=points[0]?.id;
    $('scope').textContent=`${dataset} · ${queryMode?'Query-matched evidence':'Dataset overview — search a structure to explore analogues'}`;
    const endpoint=$('endpoint').value;$('endpoint').replaceChildren();for(const name of Object.keys(points[0]?.labels||{})){const option=element('option','',name);option.value=name;$('endpoint').append(option);}if([...$('endpoint').options].some(o=>o.value===endpoint))$('endpoint').value=endpoint;
    $('endpoint').hidden=!$('endpoint').options.length;
    const scalar=points.some(p=>measurementY(p)!==null);document.querySelector('[data-view="measurements"]').disabled=!scalar;if(view==='measurements'&&!scalar)view='landscape';
    draw();renderInspector();renderTray();
  });
  new ResizeObserver(resize).observe(document.querySelector('.workbench'));
  send('streamlit:componentReady',{apiVersion:1});resize();
})();
