import multiprocessing as mp
import traceback
import os
import time

class RemoteWorker(object):
    def __init__(self) -> None:
        self.worker = None

    def setup_env(self, gpu_id, rank, world_size, port):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["LOCAL_RANK"] = str(gpu_id)
        import torch.distributed as dist
        # Use address of one of the machines
        print("==================================================")
        print(port)
        dist.init_process_group("nccl", init_method='tcp://127.0.0.1:' + str(port),
                        rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        print("Worker[" + str(rank) + "] env setup!", flush=True)

    def init_worker(self, model_config, parallel_config, scheduler_config):
        from vllm.worker.worker import Worker
        self.worker = Worker(model_config, parallel_config, scheduler_config, self.rank, None)
        print("Worker[" + str(self.rank) + "] started!", flush=True)

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def execute_method(self, method, *args, **kwargs):
        executor = getattr(self, method)
        return executor(*args, **kwargs)


def remote_worker_proc(rx, tx):
    remote_worker = RemoteWorker()
    # cmd: [quit]
    # cmd: [setup_env, gpu_id, rank, world_size, port]
    # cmd: [init_worker, model_config, parallel_config, scheduler_config]
    # cmd: [run_cmd, method, args, kwargs]
    cmd = rx.recv()
    port = 0
    while cmd[0] != "quit":
        # print(cmd)
        if cmd[0] == "setup_env":
            remote_worker.setup_env(cmd[1], cmd[2], cmd[3], cmd[4])
            port = cmd[4]
        elif cmd[0] == "init_worker":
            remote_worker.init_worker(cmd[1], cmd[2], cmd[3])
        elif cmd[0] == "execute_method":
            try:
                start_time = time.time()
                # if port == 20001:
                    # print("\nstart:", start_time, flush=True)
                res = remote_worker.execute_method(cmd[1], *cmd[2], **cmd[3])
                # print("Get result", res)
                end_time = time.time()
                tx.send((res, end_time - start_time))
                #if port == 20001:
                    # print("\nend: ", end_time, flush=True)

            except:
                traceback.print_exc()
                tx.send("ERROR!")
        cmd = rx.recv()

class PipeWorkerResult(object):
    def __init__(self, worker, res_id):
        self.worker = worker
        self.res_id = res_id
    
    def get(self):
        return self.worker.get_result(self.res_id)

    def available(self):
        return self.worker.has_result(self.res_id)

class PipeWorker(object):
    def __init__(self):
        self.rx, self.rx_child = mp.Pipe(False)
        self.tx_child, self.tx = mp.Pipe(False)
        self.max_id = 0
        self.cur_id = -1
        self.result_map = {}
        self.process = mp.Process(target=remote_worker_proc, args=(self.tx_child, self.rx_child))
        self.process.start()

    def setup_env(self, gpu_id, rank, world_size, port):
        self.tx.send(["setup_env", gpu_id, rank, world_size, port])
    
    def init_worker(self, model_config, parallel_config, scheduler_config):
        self.tx.send(["init_worker", model_config, parallel_config, scheduler_config])
    
    def run_cmd(self, cmd_name, *args, **kwargs):
        res_id = self.max_id
        self.tx.send(["execute_method", cmd_name, args, kwargs])
        self.max_id += 1
        return PipeWorkerResult(self, res_id)

    def get_result(self, res_id):
        assert res_id < self.max_id

        while self.cur_id < res_id:
            result = self.pop_result()
            self.cur_id += 1
            self.result_map[self.cur_id] = result

        result = self.result_map[res_id]
        del self.result_map[res_id]
        
        return result

    def has_result(self, res_id):
        assert res_id < self.max_id

        while self.cur_id < res_id and (self.rx.poll()):
            result = self.pop_result()
            self.cur_id += 1
            self.result_map[self.cur_id] = result

        if self.cur_id < res_id:
            return False

        return True
    
    def pop_result(self):
        return self.rx.recv()

    def stop(self):
        self.tx.send(["quit"])
