"""Three spawn workers plus the caller: dispatcher, obstacles and landing."""
from collections import defaultdict
from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty,Full
import time
import numpy as np

@dataclass
class Frame:
    sequence:int
    sample_time:float
    points:np.ndarray

def offer_latest(channel,item):
    """Bounded, nonblocking admission. Queue size is never used as synchronization."""
    try:channel.put_nowait(item);return True
    except Full:pass
    try:channel.get_nowait()
    except Empty:pass
    try:channel.put_nowait(item);return True
    except Full:return False

def _dispatch(incoming,obstacles,landing,stop):
    # Feeder threads need not flush obsolete frames during process shutdown.
    obstacles.cancel_join_thread();landing.cancel_join_thread()
    while not stop.is_set():
        try:frame=incoming.get(timeout=0.1)
        except Empty:continue
        offer_latest(obstacles,frame);offer_latest(landing,frame)

def _algorithm(name,incoming,outgoing,stop):
    from .geometry import obstacle_pipeline
    from .landing import landing_candidate
    outgoing.cancel_join_thread()
    while not stop.is_set():
        try:frame=incoming.get(timeout=0.1)
        except Empty:continue
        try:
            output=obstacle_pipeline(frame.points) if name=="obstacles" else landing_candidate(frame.points)
            result=dict(sequence=frame.sequence,sample_time=frame.sample_time,kind=name,
                        result=output,error=None,finished=time.monotonic())
        except Exception as error:
            result=dict(sequence=frame.sequence,sample_time=frame.sample_time,kind=name,
                        result=None,error=type(error).__name__+": "+str(error),finished=time.monotonic())
        offer_latest(outgoing,result)

class PerceptionPipeline:
    def __init__(self,capacity=3):
        if capacity<1:raise ValueError("invalid capacity")
        context=mp.get_context("spawn")
        self.stop=context.Event();self.queues=[context.Queue(capacity) for _ in range(4)]
        incoming,obstacles,landing,outgoing=self.queues
        self.processes=[context.Process(target=_dispatch,args=(incoming,obstacles,landing,self.stop)),
            context.Process(target=_algorithm,args=("obstacles",obstacles,outgoing,self.stop)),
            context.Process(target=_algorithm,args=("landing",landing,outgoing,self.stop))]
        self.started=False
    def start(self):
        if self.started:raise RuntimeError("pipeline already started")
        for process in self.processes:process.start()
        self.started=True
    def submit(self,frame):
        if not self.started or self.stop.is_set():return False
        return offer_latest(self.queues[0],frame)
    def poll(self,limit=16):
        results=[]
        for _ in range(limit):
            try:results.append(self.queues[-1].get_nowait())
            except Empty:break
        return results
    def close(self):
        self.stop.set()
        if self.started:
            deadline=time.monotonic()+3
            for process in self.processes:process.join(max(0,deadline-time.monotonic()))
            for process in self.processes:
                if process.is_alive():process.terminate();process.join(1)
        for channel in self.queues:channel.cancel_join_thread();channel.close()
        self.started=False

class ResultJoiner:
    def __init__(self,ttl=1.0,limit=64):self.ttl,self.limit,self.pending=ttl,limit,{}
    def add(self,result,now=None):
        now=time.monotonic() if now is None else now
        self.pending={k:v for k,v in self.pending.items() if now-v["created"]<self.ttl}
        key=result["sequence"]
        entry=self.pending.setdefault(key,{"created":now,"parts":{}})
        entry["parts"][result["kind"]]=result
        if {"obstacles","landing"}<=entry["parts"].keys():
            del self.pending[key];return entry["parts"]
        while len(self.pending)>self.limit:del self.pending[next(iter(self.pending))]
        return None
