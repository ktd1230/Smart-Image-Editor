import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataloader import (_BaseDataLoaderIter, DataLoader)
from torch._C import (_set_worker_signal_handlers, _set_worker_pids,
    _remove_worker_pids, _error_if_any_worker_fails)
import random
import threading
import sys
from torch._utils import ExceptionWrapper
from torch.utils.data._utils.pin_memory import _pin_memory_loop, pin_memory
from torch.utils.data._utils.signal_handling import _set_SIGCHLD_handler,_remove_worker_pids
from torch.utils.data import _utils
"""
A hacky way to schedule different scales with dataset.get()
"""

##### original code ######
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

##### overwrites #####
def _worker_loop(dataset, index_queue, data_queue, collate_fn, seed, init_fn, worker_id):
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    if init_fn is not None:
        init_fn(worker_id)

    while True:
        # try:
        #     r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        # except queue.Empty:
        #     break
        r = index_queue.get()
        if r is None:
            break
        idx, random_var, batch_indices = r
        try:
            samples = collate_fn([dataset.get(i, random_var) for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))
            del samples


class MyDataLoaderIter(_BaseDataLoaderIter):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.random_vars = loader.random_vars
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory and torch.cuda.is_available()
        self.timeout = loader.timeout
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            self.worker_init_fn = loader.worker_init_fn
            self.index_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
            self.worker_queue_idx = 0
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queues[i],
                          self.worker_result_queue, self.collate_fn, base_seed + i,
                          self.worker_init_fn, i))
                for i in range(self.num_workers)]

            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(self.worker_result_queue, self.data_queue, maybe_device_id, self.done_event))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()
            # 이부분 _update_worker_pids 를 _set_worker_pids로 변경
            _set_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()


    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            random_var = None
            if self.random_vars:
                random_var = random.choice(self.random_vars)
            batch = self.collate_fn([self.dataset.get(i, random_var) for i in indices])
            if self.pin_memory:
                batch = pin_memory(batch)
            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            print("확인용@@@", self._process_next_batch(batch))
            return self._process_next_batch(batch)

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        chose_var = None
        if self.random_vars:
            chose_var = random.choice(self.random_vars)
        self.index_queues[self.worker_queue_idx].put((self.send_idx, chose_var, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            # make multiline KeyError msg readable by working around
            # a python bug https://bugs.python.org/issue2651
            if batch.exc_type == KeyError and "\n" in batch.exc_msg:
                raise Exception("KeyError:" + batch.exc_msg)
            else:
                raise batch.exc_type(batch.exc_msg)
        return batch

    def _shutdown_workers(self):
        # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
        # the logic of this function.
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self.shutdown:
            self.shutdown = True
            try:
                self.done_event.set()

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, 'pin_memory_thread'):
                    # Use hasattr in case error happens before we set the attribute.
                    # First time do `worker_result_queue.put` in this process.

                    # `cancel_join_thread` in case that `pin_memory_thread` exited.
                    self.worker_result_queue.cancel_join_thread()
                    self.worker_result_queue.put(None)
                    self.pin_memory_thread.join()
                    # Indicate that no more data will be put on this queue by the
                    # current process. This **must** be called after
                    # `pin_memory_thread` is joined because that thread shares the
                    # same pipe handles with this loader thread. If the handle is
                    # closed, Py3 will error in this case, but Py2 will just time
                    # out even if there is data in the queue.
                    self.worker_result_queue.close()

                # Exit workers now.
                for q in self.index_queues:
                    q.put(None)
                    # Indicate that no more data will be put on this queue by the
                    # current process.
                    q.close()
                for w in self.workers:
                    w.join()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self.worker_pids_set:
                    _remove_worker_pids(id(self))
                    self.worker_pids_set = False

class MyDataLoader(DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, the ``shuffle`` argument is ignored.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional)
        pin_memory (bool, optional)
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
    """
    __initialized = False

    def __init__(self, dataset, random_vars=[], **kwargs):
        self.random_vars = random_vars
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        return MyDataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
