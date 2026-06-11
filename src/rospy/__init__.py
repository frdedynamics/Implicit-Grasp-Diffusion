class Publisher:
    def __init__(self, *args, **kwargs): pass
    def publish(self, *args, **kwargs): pass

class Subscriber:
    def __init__(self, *args, **kwargs): pass

class Duration:
    def __init__(self, secs=0, nsecs=0):
        self.secs = secs
        self.nsecs = nsecs
    def to_sec(self): return self.secs + self.nsecs * 1e-9

class Rate:
    def __init__(self, hz): pass
    def sleep(self): pass

def init_node(*args, **kwargs): pass
def is_shutdown(): return False
def loginfo(msg): print(f"[Fake rospy] {msg}")
def logwarn(msg): print(f"[Fake rospy] WARNING: {msg}")
def logerr(msg): print(f"[Fake rospy] ERROR: {msg}")
def sleep(duration): 
    import time
    if isinstance(duration, (int, float)): time.sleep(duration)
    else: time.sleep(getattr(duration, 'secs', 0))

def get_param(name, default=None): return default
def Time(secs=0, nsecs=0): return 0
def Service(*args, **kwargs): pass
