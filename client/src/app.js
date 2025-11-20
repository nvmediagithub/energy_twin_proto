import {genGeo, buildGrid, cmd, simStart, simPause, simStep, openWS} from "./api.js";

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
let W=0,H=0;
let geo=null, grid=null;
let selected=null;
let tool="select";
let connectFirst=null;

function resize(){
  W = canvas.width = canvas.clientWidth * devicePixelRatio;
  H = canvas.height = canvas.clientHeight * devicePixelRatio;
  draw();
}
window.addEventListener("resize", resize);
resize();

function worldToScreen(p){
  return {x:(p.x/geo.width)*W, y:(p.y/geo.height)*H};
}
function screenToWorld(x,y){
  return {x:(x/W)*geo.width, y:(y/H)*geo.height};
}

function pointToSegmentDistance(px, py, x1, y1, x2, y2){
  const dx=x2-x1; const dy=y2-y1;
  if(dx===0 && dy===0){
    return Math.hypot(px-x1, py-y1);
  }
  const t=((px-x1)*dx + (py-y1)*dy)/(dx*dx+dy*dy);
  const clamped=Math.max(0, Math.min(1, t));
  const cx=x1 + clamped*dx;
  const cy=y1 + clamped*dy;
  return Math.hypot(px-cx, py-cy);
}

function resolveSelectionInGrid(newGrid, prev){
  if(!prev || !newGrid) return null;
  if(prev.type==="line"){
    return newGrid.edges.find(e=>e.id===prev.id)||null;
  }
  return newGrid.nodes.find(n=>n.id===prev.id)||null;
}

async function updateNodeProps(changes){
  if(!grid || !selected || selected.type==="line") return;
  const prev=selected;
  grid = await cmd(grid.id,"update_props",{node_id:selected.id, props:changes});
  selected = resolveSelectionInGrid(grid, prev);
  renderPanel(); draw();
}

async function updateLineStatus(status){
  if(!grid || !selected || selected.type!=="line") return;
  const prev=selected;
  grid = await cmd(grid.id,"set_line_status",{line_id:selected.id, status});
  selected = resolveSelectionInGrid(grid, prev);
  renderPanel(); draw();
}

function draw(){
  if(!geo){ ctx.clearRect(0,0,W,H); return; }
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle="#b8c9ad"; ctx.fillRect(0,0,W,H);

  if(geo.river){
    const pts=geo.river.points;
    ctx.strokeStyle="#87b7c8";
    ctx.lineWidth=geo.river.width*(W/geo.width);
    ctx.lineCap="round"; ctx.lineJoin="round";
    ctx.beginPath();
    pts.forEach((p,i)=>{ const s=worldToScreen(p); i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y); });
    ctx.stroke();
  }

  ctx.lineCap="round"; ctx.lineJoin="round";
  geo.roads.forEach(r=>{
    const pts=r.polyline;
    const w1=(r.kind==="major"?3.2:2.0)*(W/geo.width);
    ctx.strokeStyle="#e6decf"; ctx.lineWidth=w1*1.8;
    ctx.beginPath(); pts.forEach((p,i)=>{ const s=worldToScreen(p); i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y); }); ctx.stroke();
    ctx.strokeStyle="#6f675f"; ctx.lineWidth=w1;
    ctx.beginPath(); pts.forEach((p,i)=>{ const s=worldToScreen(p); i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y); }); ctx.stroke();
  });

  geo.houses.forEach(h=>{
    const s=worldToScreen(h);
    ctx.fillStyle="#c59a7a";
    ctx.beginPath(); ctx.arc(s.x,s.y,2.2*(W/geo.width),0,Math.PI*2); ctx.fill();
  });

  if(!grid) return;

  const zones=grid.overlays?.pole_load_zones||[];
  zones.forEach(z=>{
    const s=worldToScreen(z);
    const rad=z.radius*(W/geo.width);
    const col=z.color==="green"?"rgba(0,200,100,0.18)":
              z.color==="yellow"?"rgba(240,200,60,0.18)":"rgba(240,60,60,0.20)";
    ctx.fillStyle=col;
    ctx.beginPath(); ctx.arc(s.x,s.y,rad,0,Math.PI*2); ctx.fill();
  });

  grid.edges.forEach(e=>{
    const a=grid.nodes.find(n=>n.id===e.from_id);
    const b=grid.nodes.find(n=>n.id===e.to_id);
    if(!a||!b) return;
    const sa=worldToScreen(a), sb=worldToScreen(b);
    const baseWidth=1.2*(W/geo.width);
    let stroke="#2d2a27";
    let lineWidth=baseWidth;
    ctx.setLineDash([]);
    if(e.status==="open"){
      stroke="#7a7671";
      ctx.setLineDash([6*devicePixelRatio,6*devicePixelRatio]);
    } else if(e.status==="faulted"){
      stroke="#c32c2c";
      lineWidth=baseWidth*1.8;
    }
    ctx.strokeStyle=stroke;
    ctx.lineWidth=lineWidth;
    ctx.beginPath(); ctx.moveTo(sa.x,sa.y); ctx.lineTo(sb.x,sb.y); ctx.stroke();

    if(e.status==="faulted"){
      const mx=(sa.x+sb.x)/2, my=(sa.y+sb.y)/2;
      const len=6*(W/geo.width);
      ctx.setLineDash([]);
      ctx.strokeStyle="#fff";
      ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(mx-len,my-len); ctx.lineTo(mx+len,my+len);
      ctx.moveTo(mx-len,my+len); ctx.lineTo(mx+len,my-len);
      ctx.stroke();
    }

    if(selected && selected.type==="line" && selected.id===e.id){
      ctx.setLineDash([]);
      ctx.strokeStyle="rgba(255,255,255,0.6)";
      ctx.lineWidth=lineWidth + 1.5*(W/geo.width);
      ctx.beginPath(); ctx.moveTo(sa.x,sa.y); ctx.lineTo(sb.x,sb.y); ctx.stroke();
    }
  });

  grid.nodes.forEach(n=>{
    const s=worldToScreen(n);
    let r=3.5, fill="#444";
    if(n.type==="source"){ r=6; fill="#333"; }
    else if(n.type==="pole"){ r=3.5; fill="#222"; }
    else if(n.type==="consumer"){ r=2.5; fill="#555"; }
    ctx.fillStyle=fill;

    if(selected && selected.id===n.id){
      ctx.strokeStyle="#fff"; ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(s.x,s.y,(r+2)*(W/geo.width),0,Math.PI*2); ctx.stroke();
    }
    ctx.beginPath(); ctx.arc(s.x,s.y,r*(W/geo.width),0,Math.PI*2); ctx.fill();
  });
}

function hitTest(sx,sy){
  if(!grid) return null;
  let bestNode=null, nodeDist=1e9;
  grid.nodes.forEach(n=>{
    const s=worldToScreen(n);
    const d=Math.hypot(s.x-sx, s.y-sy);
    if(d<nodeDist){ nodeDist=d; bestNode=n; }
  });
  const threshold=10*devicePixelRatio;
  if(bestNode && nodeDist<threshold){
    return bestNode;
  }
  let bestEdge=null, edgeDist=1e9;
  grid.edges.forEach(e=>{
    const a=grid.nodes.find(n=>n.id===e.from_id);
    const b=grid.nodes.find(n=>n.id===e.to_id);
    if(!a||!b) return;
    const sa=worldToScreen(a), sb=worldToScreen(b);
    const dist=pointToSegmentDistance(sx,sy,sa.x,sa.y,sb.x,sb.y);
    if(dist<edgeDist){ edgeDist=dist; bestEdge=e; }
  });
  if(bestEdge && edgeDist<threshold){
    return bestEdge;
  }
  return null;
}

canvas.addEventListener("click", async (ev)=>{
  if(!geo) return;
  const rect=canvas.getBoundingClientRect();
  const sx=(ev.clientX-rect.left)*devicePixelRatio;
  const sy=(ev.clientY-rect.top)*devicePixelRatio;
  const wp=screenToWorld(sx,sy);

  if(tool==="select"){
    selected=hitTest(sx,sy);
    renderPanel(); draw(); return;
  }
  if(tool==="add_source"){
    grid=await cmd(grid.id,"add_source",{position:wp}); draw(); return;
  }
  if(tool==="add_pole"){
    grid=await cmd(grid.id,"add_pole",{position:wp}); draw(); return;
  }
  if(tool==="connect_line"){
    const ht=hitTest(sx,sy);
    if(!ht) return;
    if(!connectFirst){ connectFirst=ht; hint(`first: ${ht.id}`); }
    else{
      grid=await cmd(grid.id,"connect_line",{from_id:connectFirst.id,to_id:ht.id});
      connectFirst=null; hint(""); draw();
    }
  }
});

function renderPanel(){
  const selDiv=document.getElementById("sel");
  const panel=document.getElementById("panel");
  panel.innerHTML="";
  if(!selected){ selDiv.textContent="None"; return; }
  selDiv.textContent=`${selected.type} (${selected.id})`;
  const props=selected.props||{}, state=selected.state||{};

  const infoBlock=document.createElement("div");
  infoBlock.style.marginBottom="10px";
  const statusLine=document.createElement("div");
  statusLine.textContent=`status: ${selected.status||"unknown"}`;
  infoBlock.appendChild(statusLine);

  const appendStateLine=(label,value)=>{
    if(value===undefined||value===null) return;
    const row=document.createElement("div");
    const formatted=typeof value==="number"?value.toFixed(2):value;
    row.textContent=`${label}: ${formatted}`;
    infoBlock.appendChild(row);
  };
  const stateFields=[
    {label:"P (kW)", key:"p_kw"},
    {label:"Q (kvar)", key:"q_kvar"},
    {label:"S (kVA)", key:"s_kva"},
    {label:"P flow (kW)", key:"p_flow_kw"},
    {label:"Q flow (kvar)", key:"q_flow_kvar"},
    {label:"S flow (kVA)", key:"s_flow_kva"},
    {label:"P load (kW)", key:"p_load_kw"},
    {label:"Q load (kvar)", key:"q_load_kvar"},
    {label:"S load (kVA)", key:"s_load_kva"},
    {label:"V (kV)", key:"v_kv"},
    {label:"Island", key:"island_id"},
    {label:"Overload", key:"overload_ratio"},
    {label:"Supplied", key:"supplied"},
  ];
  stateFields.forEach(f=>appendStateLine(f.label,state[f.key]));
  if(selected.type==="consumer"){
    const profileState=props.profile||"residential";
    appendStateLine("Profile", profileState);
    const cosPhi=props.cos_phi;
    if(cosPhi!==undefined) appendStateLine("cosφ", cosPhi);
  }
  panel.appendChild(infoBlock);

  if(selected.type==="consumer"){
    const profileRow=document.createElement("div"); profileRow.className="row";
    const profileLabel=document.createElement("label"); profileLabel.textContent="Profile";
    const profileSelect=document.createElement("select");
    ["residential","commercial","industrial","nightlife"].forEach(opt=>{
      const o=document.createElement("option"); o.value=opt; o.textContent=opt;
      if(props.profile===opt) o.selected=true;
      profileSelect.appendChild(o);
    });
    profileSelect.onchange=()=>updateNodeProps({profile:profileSelect.value});
    profileRow.appendChild(profileLabel); profileRow.appendChild(profileSelect);
    panel.appendChild(profileRow);

    const cosRow=document.createElement("div"); cosRow.className="row";
    const cosLabel=document.createElement("label"); cosLabel.textContent="cosφ (0.5-1.0)";
    const cosInput=document.createElement("input");
    cosInput.type="number"; cosInput.step="0.01"; cosInput.min="0.5"; cosInput.max="1.0";
    cosInput.value=(props.cos_phi!==undefined?props.cos_phi:0.95);
    cosInput.onchange=()=>updateNodeProps({cos_phi:parseFloat(cosInput.value)});
    cosRow.appendChild(cosLabel); cosRow.appendChild(cosInput);
    panel.appendChild(cosRow);
  }

  if(selected.type==="line"){
    const statusRow=document.createElement("div"); statusRow.className="row";
    const statusLabel=document.createElement("label"); statusLabel.textContent="Status";
    const statusSelect=document.createElement("select");
    ["online","open","faulted"].forEach(opt=>{
      const o=document.createElement("option"); o.value=opt; o.textContent=opt;
      if(selected.status===opt) o.selected=true;
      statusSelect.appendChild(o);
    });
    statusSelect.onchange=()=>updateLineStatus(statusSelect.value);
    statusRow.appendChild(statusLabel); statusRow.appendChild(statusSelect);
    panel.appendChild(statusRow);
  } else {
    const editableKeys=Object.keys(props).filter(k=>!(selected.type==="consumer"&&(k==="profile"||k==="cos_phi")));
    editableKeys.forEach(k=>{
      const wrap=document.createElement("div"); wrap.className="row";
      const lab=document.createElement("label"); lab.textContent=k;
      const inp=document.createElement("input");
      inp.value=props[k];
      inp.onchange=()=>
        updateNodeProps({[k]:isNaN(Number(inp.value))?inp.value:Number(inp.value)});
      wrap.appendChild(lab); wrap.appendChild(inp); panel.appendChild(wrap);
    });
  }

  const pre=document.createElement("pre");
  pre.style.whiteSpace="pre-wrap"; pre.style.fontSize="12px";
  pre.textContent="state\n"+JSON.stringify(state,null,2);
  panel.appendChild(pre);
}

function hint(t){ document.getElementById("hint").textContent=t; }
document.getElementById("tool").onchange=e=>{ tool=e.target.value; connectFirst=null; hint(""); };

document.getElementById("gen").onclick=async ()=>{
  geo=await genGeo(+seed.value, tags.value);
  grid=null; selected=null; renderPanel(); draw();
};
document.getElementById("build").onclick=async ()=>{
  if(!geo) return; grid=await buildGrid(geo.id); selected=null; renderPanel(); draw();
};
document.getElementById("start").onclick=async ()=>{
  if(!grid) return; await simStart(grid.id, +dt.value);
};
document.getElementById("pause").onclick=async ()=>{ await simPause(); };
document.getElementById("step").onclick=async ()=>{ if(grid) await simStep(1); };
document.getElementById("triggerFault")?.addEventListener("click", async ()=>{
  if(!grid) return;
  const prev=selected;
  grid=await cmd(grid.id,"trigger_fault",{});
  selected=resolveSelectionInGrid(grid, prev);
  renderPanel(); draw();
});
document.getElementById("clearFaults")?.addEventListener("click", async ()=>{
  if(!grid) return;
  const prev=selected;
  grid=await cmd(grid.id,"clear_faults",{});
  selected=resolveSelectionInGrid(grid, prev);
  renderPanel(); draw();
});

openWS((snap)=>{
  if(grid && snap.id===grid.id){
    const prev=selected;
    grid=snap;
    selected=resolveSelectionInGrid(grid, prev);
    renderPanel(); draw();
  }
});

draw();
