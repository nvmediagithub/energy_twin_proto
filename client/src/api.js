const base = "";

export async function genGeo(seed, tags){
  const r = await fetch(`${base}/geo/generate?seed=${seed}&width=300&height=300&tags=${encodeURIComponent(tags)}`, {method:"POST"});
  return r.json();
}
export async function buildGrid(geo_id){
  const r = await fetch(`${base}/grid/build_from_geo?geo_id=${geo_id}`, {method:"POST"});
  return r.json();
}
export async function cmd(grid_id, type, payload){
  const r = await fetch(`${base}/grid/${grid_id}/command`, {
    method:"PATCH",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({type, payload})
  });
  return r.json();
}
export async function simStart(grid_id, dt){
  const r = await fetch(`${base}/sim/start?grid_id=${grid_id}&dt=${dt}`, {method:"POST"});
  return r.json();
}
export async function simPause(){
  const r = await fetch(`${base}/sim/pause`, {method:"POST"}); return r.json();
}
export async function simStep(n=1){
  const r = await fetch(`${base}/sim/step?n=${n}`, {method:"POST"}); return r.json();
}
export function openWS(onMsg){
  const ws = new WebSocket(`ws://${location.host}/sim/stream`);
  ws.onopen=()=>ws.send("hi");
  ws.onmessage=(ev)=>onMsg(JSON.parse(ev.data));
  return ws;
}
