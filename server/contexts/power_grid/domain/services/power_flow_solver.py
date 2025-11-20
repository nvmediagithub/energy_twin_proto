from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
import heapq, math
from ..grid import Grid
from ..entities import Transformer, Pole, Consumer, Line, Source

class RadialPQDistFlowSolver:
    """
    Упрощённый радиальный P/Q DistFlow.
    - Использует только линии status=online.
    - Строит лес от источников (multi-source Dijkstra).
    - Считает P/Q назад, напряжения вперёд.
    - Недостижимые узлы -> island (v=0).
    """

    def solve(self, g: Grid) -> Grid:
        sources: List[Source] = g.sources()
        if not sources:
            self._deenergize_all(g)
            return g

        # adjacency only for online lines
        adj: Dict[str, List[Tuple[str, str, float]]] = {nid: [] for nid in g.nodes}
        for ln in g.lines.values():
            if ln.status != "online":
                continue
            a, b = ln.from_id, ln.to_id
            length = float(ln.props.get("length", 1.0))
            adj[a].append((b, ln.id, length))
            adj[b].append((a, ln.id, length))

        # multi-source Dijkstra -> parent forest
        dist = {nid: math.inf for nid in g.nodes}
        parent: Dict[str, Optional[str]] = {nid: None for nid in g.nodes}
        parent_line: Dict[str, Optional[str]] = {nid: None for nid in g.nodes}
        pq=[]
        for s in sources:
            dist[s.id]=0.0
            heapq.heappush(pq,(0.0, s.id))

        while pq:
            d,u=heapq.heappop(pq)
            if d!=dist[u]:
                continue
            for v,lid,w in adj.get(u,[]):
                nd=d+w
                if nd<dist[v]:
                    dist[v]=nd
                    parent[v]=u
                    parent_line[v]=lid
                    heapq.heappush(pq,(nd,v))

        # reachable set (energized)
        reachable: Set[str] = {nid for nid, d in dist.items() if math.isfinite(d)}

        # children map
        children={nid:[] for nid in g.nodes}
        for nid,pid in parent.items():
            if pid is not None:
                children[pid].append(nid)

        # nodal injections P/Q (consumers)
        node_p={nid:0.0 for nid in g.nodes}
        node_q={nid:0.0 for nid in g.nodes}
        for n in g.nodes.values():
            if isinstance(n, Consumer):
                node_p[n.id]=float(n.state.get("p_kw", n.state.get("load_kw", n.props.get("base_kw",1.5))))
                node_q[n.id]=float(n.state.get("q_kvar", 0.0))

        # backward sweep P/Q
        order=sorted(g.nodes.keys(), key=lambda nid: dist[nid], reverse=True)
        flow_p={nid:node_p[nid] for nid in g.nodes}
        flow_q={nid:node_q[nid] for nid in g.nodes}
        for nid in order:
            pid=parent[nid]
            if pid is not None:
                flow_p[pid]+=flow_p[nid]
                flow_q[pid]+=flow_q[nid]

        # seed sources
        for s in sources:
            v0=float(s.props.get("voltage_out_kv",10.0))
            s.state["v_kv"]=v0
            s.state["p_out_kw"]=flow_p[s.id]
            s.state["q_out_kvar"]=flow_q[s.id]

        # forward sweep
        order_fwd=sorted(g.nodes.keys(), key=lambda nid: dist[nid])
        for nid in order_fwd:
            if nid not in reachable:
                continue
            n=g.nodes[nid]
            v_parent=float(n.state.get("v_kv", float(n.props.get("voltage_kv",10.0))))

            for cid in children.get(nid,[]):
                if cid not in reachable:
                    continue
                cnode=g.nodes[cid]
                lid=parent_line[cid]
                if not lid:
                    continue
                ln: Line = g.lines[lid]

                length=float(ln.props.get("length",1.0))
                cap=float(ln.props.get("capacity_kva",
                                       ln.props.get("capacity_kw",200.0)))
                r_pu=float(ln.props.get("r_pu",0.0015))
                x_pu=float(ln.props.get("x_pu",0.0025))

                pflow=flow_p[cid]
                qflow=flow_q[cid]
                sflow=math.sqrt(pflow*pflow + qflow*qflow)

                # simplified PQ voltage drop
                dv=(r_pu*pflow + x_pu*qflow)*length/max(v_parent,0.1)
                v_child=max(0.0, v_parent-dv)

                ln.state["p_flow_kw"]=pflow
                ln.state["q_flow_kvar"]=qflow
                ln.state["s_flow_kva"]=sflow
                ln.state["loss_kw"]=r_pu*(sflow**2)*length*0.0005
                ln.state["overload_ratio"]=sflow/cap if cap>1e-9 else 0.0

                if isinstance(cnode, Transformer):
                    vout=float(cnode.props.get("voltage_out_kv",0.4))
                    eff=float(cnode.props.get("efficiency",0.98))
                    ccap=float(cnode.props.get("capacity_kva",250.0))

                    cnode.state["v_kv"]=vout
                    cnode.state["p_in_kw"]=pflow
                    cnode.state["q_in_kvar"]=qflow
                    cnode.state["p_out_kw"]=pflow*eff
                    cnode.state["q_out_kvar"]=qflow*eff
                    cnode.state["overload_ratio"]=math.sqrt(
                        cnode.state["p_out_kw"]**2 + cnode.state["q_out_kvar"]**2
                    )/ccap if ccap>1e-9 else 0.0
                else:
                    cnode.state["v_kv"]=v_child

        # poles state
        for n in g.nodes.values():
            if isinstance(n, Pole):
                if n.id in reachable:
                    p=flow_p[n.id]; q=flow_q[n.id]; s=math.sqrt(p*p+q*q)
                    n.state["p_load_kw"]=p
                    n.state["q_load_kvar"]=q
                    n.state["s_load_kva"]=s
                    cap=float(n.props.get("capacity_kw",60.0))
                    n.state["overload_ratio"]=s/cap if cap>1e-9 else 0.0
                else:
                    self._deenergize_node(n)

        # consumers supplied/islands
        island_id=0
        for nid in g.nodes.keys():
            n=g.nodes[nid]
            if nid in reachable:
                n.state["island_id"]=0
                if isinstance(n, Consumer):
                    v=float(n.state.get("v_kv",0.0))
                    n.state["supplied"]=v>0.2
            else:
                island_id=1
                n.state["island_id"]=island_id
                self._deenergize_node(n)

        # deenergize lines not online or not in reachable forest
        for ln in g.lines.values():
            if ln.status != "online":
                ln.state.setdefault("overload_ratio",0.0)
                ln.state["p_flow_kw"]=0.0
                ln.state["q_flow_kvar"]=0.0
                ln.state["s_flow_kva"]=0.0
                continue
            if ln.from_id not in reachable or ln.to_id not in reachable:
                ln.state["p_flow_kw"]=0.0
                ln.state["q_flow_kvar"]=0.0
                ln.state["s_flow_kva"]=0.0
                ln.state["overload_ratio"]=0.0

        return g

    def _deenergize_all(self, g: Grid):
        for n in g.nodes.values():
            self._deenergize_node(n)
        for ln in g.lines.values():
            ln.state["p_flow_kw"]=0.0
            ln.state["q_flow_kvar"]=0.0
            ln.state["s_flow_kva"]=0.0
            ln.state["overload_ratio"]=0.0

    def _deenergize_node(self, n):
        n.state["v_kv"]=0.0
        if isinstance(n, Consumer):
            n.state["supplied"]=False
            n.state["p_kw"]=0.0
            n.state["q_kvar"]=0.0
            n.state["s_kva"]=0.0
        if isinstance(n, Pole):
            n.state["p_load_kw"]=0.0
            n.state["q_load_kvar"]=0.0
            n.state["s_load_kva"]=0.0
            n.state["overload_ratio"]=0.0
