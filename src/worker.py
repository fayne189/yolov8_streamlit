import queue
import sys
import threading
import traceback
from typing import Optional

from .logger import get_logger

# sys.path.append('..')

# from logger import get_logger

SENTINEL = "__SENTINEL__"

logger = get_logger(__name__)

class Worker(object):

    def __init__(
        self,
        stop_timeout: Optional[float] = None,
    ):
        super().__init__()  # don't forget this!

        self._thread = threading.Thread(target=self._run_worker_thread)
        self._in_queue = queue.Queue()
        self._latest_result_lock = threading.Lock()
        self._out_queue = None
        self._thread.start()

        self.stop_timeout = stop_timeout

    def _run_worker_thread(self):
        try:
            self._worker_thread()
        except Exception:
            logger.error("Error occurred in the worker thread:")

            exc_type, exc_value, exc_traceback = sys.exc_info()
            for tb in traceback.format_exception(exc_type, exc_value, exc_traceback):
                for tbline in tb.rstrip().splitlines():
                    logger.error(tbline.rstrip())

    def _worker_thread(self):
        while True:
            item = self._in_queue.get()
            if item == SENTINEL:
                break

            stop_requested = False
            while not self._in_queue.empty():
                item = self._in_queue.get_nowait()
                if item == SENTINEL:
                    stop_requested = True
            if stop_requested:
                break

            if item is None:
                raise Exception("A queued item is unexpectedly None")

            result_item = self.transform(item)
            with self._latest_result_lock:
                self._latest_result_item = result_item
            if self._out_queue:
                if not self._out_queue.full():
                    self._out_queue.put_nowait(result_item)
                else:
                    raise Exception("output queue is full")

    def stop(self):
        self._in_queue.put(SENTINEL)
        self._thread.join(self.stop_timeout)
        return 

    def link(self, out_queue: queue.Queue):
        self._out_queue = out_queue
        
    def transform(self, item):
        # do something here
        return item
    
    def recv(self, item):
        self._in_queue.put(item)
        
    def get_result(self):
        with self._latest_result_lock:
            return self._latest_result_item


class Sender(Worker):
    def __init__(self, stop_timeout: float | None = None):
        super().__init__(stop_timeout)
        self.queue_dict = {}
        
    def transform(self, item):
        to_queue = self.queue_dict.get(item[1])
        to_queue.put(item[0])
        return 
    
    def update_queue_dict(self, queue_dict):
        self.queue_dict.update(queue_dict)
        
    def recv(self, item, to_queue_name):
        return super().recv((item,to_queue_name))