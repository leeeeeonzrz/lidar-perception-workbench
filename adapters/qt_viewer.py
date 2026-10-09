"""Synthetic Qt/OpenGL front end. Result polling never waits for a matching worker."""
import sys
import time
import numpy as np
from PyQt5 import QtCore,QtWidgets
import pyqtgraph.opengl as gl
from lidar_workbench.pipeline import Frame,PerceptionPipeline,ResultJoiner
from lidar_workbench.synthetic import scene

class Viewer(gl.GLViewWidget):
    def __init__(self):
        super().__init__();self.setWindowTitle("LiDAR geometry workbench — synthetic")
        self.setCameraPosition(distance=14)
        self.scatter=gl.GLScatterPlotItem(pos=np.empty((0,3)),size=2)
        self.ring=gl.GLLinePlotItem(pos=np.empty((0,3)),color=(0,1,0,1),width=3)
        self.addItem(self.scatter);self.addItem(self.ring)
        self.pipeline=PerceptionPipeline();self.pipeline.start();self.joiner=ResultJoiner()
        self.sequence=0;self.last_submit=0
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(30)
    def tick(self):
        now=time.monotonic()
        if now-self.last_submit>0.5:
            self.pipeline.submit(Frame(self.sequence,now,scene(self.sequence)))
            self.sequence+=1;self.last_submit=now
        for result in self.pipeline.poll():
            pair=self.joiner.add(result)
            if pair is None:continue
            if any(part["error"] for part in pair.values()):
                self.setWindowTitle("A worker reported an error");continue
            obstacles=pair["obstacles"]["result"]
            self.scatter.setData(pos=obstacles["points"],color=(1,0.6,0.1,1))
            candidate=pair["landing"]["result"]
            if candidate is None:self.ring.setData(pos=np.empty((0,3)))
            else:
                angles=np.linspace(0,2*np.pi,80);center=np.array(candidate["center"])
                ring=center+np.column_stack([np.cos(angles)*candidate["radius"],
                    np.sin(angles)*candidate["radius"],np.zeros_like(angles)])
                self.ring.setData(pos=ring)
    def closeEvent(self,event):
        self.timer.stop();self.pipeline.close();super().closeEvent(event)

if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv);window=Viewer();window.resize(900,600)
    window.show();sys.exit(app.exec())
