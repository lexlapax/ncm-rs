"""Instrumented stage spans; intentionally separate from headline API timings."""
import torch


class Events:
    def __init__(self):
        self.starts = {}
        self.pairs = {}

    def start(self, name):
        if name in self.starts: raise ValueError("duplicate stage start")
        event = torch.cuda.Event(enable_timing=True); event.record()
        self.starts[name] = event

    def end(self, name):
        event = torch.cuda.Event(enable_timing=True); event.record()
        self.pairs[name] = self.starts.pop(name),event


def profile(codec, tokens, batch, encode, decode):
    trials = {}
    names = ("_down_for_quant", "_distances", "_apply_quant_resi")
    if any(name in codec.vq.__dict__ for name in names):
        raise ValueError("unexpected preexisting observers")
    down,distances,phi = (getattr(codec.vq,name) for name in names)
    state = {}

    def observed_down(x, level, length):
        if state["encode"]: state["events"].start(f"encode.level_{level:02}")
        return down(x,level,length)

    def observed_distances(x, weight):
        level = state["level"]
        if level+1 == len(codec.vq.levels): state["events"].start(f"encode.level_{level:02}")
        state["level"] += 1
        return distances(x,weight)

    def observed_phi(x, level):
        value = phi(x,level)
        if state["encode"]: state["events"].end(f"encode.level_{level:02}")
        return value

    codec.vq._down_for_quant = observed_down
    codec.vq._distances = observed_distances
    codec.vq._apply_quant_resi = observed_phi
    try:
        for trial in range(35):
            ids = torch.tensor(tokens,dtype=torch.int64,device="cuda").reshape(batch,89)
            torch.cuda.synchronize()
            events = Events(); state.update(encode=True,level=0,events=events)
            codes,encoded_snapshots = encode(codec,ids,events)
            state["encode"] = False
            output = decode(codec,codes,events)
            torch.cuda.synchronize()
            if events.starts or len(events.pairs)!=15:
                raise ValueError("incomplete stage capture")
            for name,(start,end) in events.pairs.items():
                if trial>=5: trials.setdefault(name,[]).append(start.elapsed_time(end))
            del ids,codes,encoded_snapshots,output
    finally:
        for name in names: delattr(codec.vq,name)
    return trials
