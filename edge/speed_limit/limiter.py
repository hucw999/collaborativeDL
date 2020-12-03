from subprocess import Popen


class SpeedLimiter:

    def __init__(self, device='eth0'):
        self._current_limit = None
        self.device = device

    @property
    def current_limit(self):
        return self._current_limit

    @current_limit.setter
    def current_limit(self, current_limit: str):
        try:
            if current_limit is None:
                Popen(f"wondershaper clear {self.device}".split()).wait()
            elif current_limit == 'low':
                Popen(f"wondershaper {self.device} 19 19".split()).wait()
            elif current_limit == 'high':
                Popen(f"wondershaper {self.device} 20480 20480".split()).wait()
            self._current_limit = current_limit
        except FileNotFoundError:
            pass
