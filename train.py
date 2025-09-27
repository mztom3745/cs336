#from cs336_basics.pretokenization_example import find_chunk_boundaries,find_chunk_boundaries_debug
from cs336_basics.pretokenization_example import find_chunk_boundaries,find_chunk_boundaries_debug
import regex as re
from collections import defaultdict
#from cs336_basics.BPE import BPETokenizer,BPETokenizerParams,get_compression_ratio
from cs336_basics.BPE import BPETokenizer,BPETokenizerParams,get_compression_ratio
import numpy as np

# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re
from collections import defaultdict,Counter
import multiprocessing as mp
from tqdm import tqdm


def choose_dtype(vocab_size: int):
    """根据 vocab_size 自动选择最合适的 dtype"""
    if vocab_size <= 65535:
        return np.uint16
    else:
        return np.int32

def worker(
        idx_type,
        task_queue,
        result_queue,
        input_path:str,
        start:int,
        end:int,
        special_tokens: list[str] = None
):
    process_name = mp.current_process().name
    #print(f"[子进程 {process_name}] 预分词开始")
    # 比如：预分词、统计merge_hash、构造pretoken_list
    
    with open(input_path,"rb") as f:
        f.seek(start)
        data = f.read() if end is None else f.read(end - start)
        all_chunks = data.decode("utf-8", errors="ignore")
        #标准化换行
        all_chunks = all_chunks.replace("\r\n", "\n").replace("\r", "\n")
   
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        #!!
        #chunks = re.split(pattern, all_chunks)#去掉了special_token的大chunk
        chunks = [c for c in re.split(pattern, all_chunks) if c]
        #为什么一定要去掉空的：
    else:
        chunks = [all_chunks]

    local_merge_hash = Counter() 
    local_occ = defaultdict(lambda: defaultdict(list))

    pretoken_nps = []
    for chunk_id, chunk in enumerate(
        tqdm(chunks, desc=f"预分词 {process_name}", position=int(process_name.split('-')[-1]) - 1, leave=False)
    ):
        #循环每个去掉了special_token的大chunk
        #（直接不给空的就行了）哪怕chunk为空也要继续执行，因为chunk_id在递增，需要添加一个空的pretoken        
        pretokens  = []
        for pretoken_id,match in enumerate(re.finditer(PAT, chunk)):#对每个chunk进行分词
            # 用 np.frombuffer 直接转为 uint8 数组，避免中间 Python list
            # pretoken = match.group(0) #分词后不再进行跨pretoken的统计
            # pretoken = list(map(int, pretoken.encode("utf-8")))
            pretoken = np.frombuffer(match.group(0).encode("utf-8"), dtype=np.uint8).astype(idx_type, copy=True)#使用int32或者int16
            pretokens.append(pretoken)
            for i, (index1, index2) in enumerate(zip(pretoken, pretoken[1:])):
                pair = (index1, index2)
                local_merge_hash[pair] += 1
                local_occ[pair][(chunk_id,pretoken_id)].append(i)
        pretoken_nps.append(pretokens)
    #得到pretoken_list，local_merge_hash，local_occ
    
    #print(f"[子进程 {process_name}]得到local_merge_hash{local_merge_hash}")
    result_queue.put(local_merge_hash)
    print(f"[子进程 {process_name}] 完成发送local_merge_hash")
    #子进程处理循环
    while True:
        task = task_queue.get()
        if task is None:
            print(f"子进程[{process_name}] 收到退出信号，退出")
            break
        maxpair,next_idx = task
        chunk_dict = local_occ.get(maxpair)
        if chunk_dict is None:
            print(f"子进程 [{process_name}] 没有找到pair:{maxpair}")
            result_queue.put(None)
            continue
        #删-改-增逻辑
        #删除改tokenn里所有的occ（maxpair）记录以及merge_hash中的键值
        for (chunk_id, pretoken_id), pos_list in list(chunk_dict.items()):
            #之前报错是因为跳过了空的chunk
            if chunk_id >= len(pretoken_nps):
                print(f"[子进程[{process_name}]错误] chunk_id={chunk_id} 超出 pretoken_nps 长度 {len(pretoken_nps)}")
            elif pretoken_id >= len(pretoken_nps[chunk_id]):
                print(f"[子进程[{process_name}]错误] pretoken_id={pretoken_id} 超出 pretoken_nps[{chunk_id}] 长度 {len(pretoken_nps[chunk_id])}")
            token = pretoken_nps[chunk_id][pretoken_id]
            token = token.tolist()

            pos_list = sorted(pos_list,reverse=True)
              
            #不仅删maxpair,还要删所有的pair
            #local_occ[maxpair].pop((chunk_id, pretoken_id))
            

            pairs = list(zip(token[:-1], token[1:]))
            local_merge_hash.subtract(pairs)
            local_merge_hash += Counter()  # 清理掉 <=0 的项

            #删除这个token中所有pair的记录，具体是删除occ中pair到这个token的所有字典值
            for a, b in zip(token[:-1], token[1:]):
                pair_dict = local_occ.get((a, b))#(a,b)在token中或许会反复出现，所以可能之前已经删除
                if pair_dict:
                    pair_dict.pop((chunk_id, pretoken_id), None)
                    if not pair_dict:  # 如果这个pair没有任何位置记录了
                        local_occ.pop((a, b), None)

            for pos in pos_list:
                token[pos:pos+2] = [next_idx]
            pretoken_nps[chunk_id][pretoken_id] = np.array(token, dtype=idx_type)

            for newpos , (a,b) in enumerate(zip(token[:-1],token[1:])): 
                local_merge_hash[(a,b)]+=1
                local_occ[(a,b)][(chunk_id, pretoken_id)].append(newpos)
        result_queue.put(local_merge_hash)
        local_occ.pop(maxpair, None)

        #local_merge_hash.pop(maxpair, None)

def train_bpe(
    input_path:str,
    vocab_size: int,
    special_tokens: list[str] = None   
):
    #准备四个大段
    with open(input_path, "rb") as f:
        desired_num_chunks = 4
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        ranges = list(zip(boundaries[:-1],boundaries[1:]))
    idx_type = choose_dtype(vocab_size)
    num_processes = len(boundaries)-1
    print(f"一共启动{num_processes}个进程")
    #准备主进程变量
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_idx = 256
    merges: list[tuple[int, int]] = []
    special_tokens = special_tokens or []

    #共享merge_hash
    merge_hash = Counter()
    #occ = defaultdict(lambda: defaultdict(list))
    #准备多进程用的队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    processes = []
    # def worker(
        # task_queue,
        # result_queue
        # input_path:str,
        # start:int,
        # end:int,
        # special_tokens: list[str] = None
    # ):
    for start, end in ranges:
        p = mp.Process(target=worker, args=(idx_type,task_queue, result_queue, input_path, start, end, special_tokens))
        p.start()
        processes.append(p)

    for _ in range(num_processes):
        local_merge_hash = result_queue.get()
        if local_merge_hash is None:
            continue
        merge_hash.update(local_merge_hash)
    #print(f"主进程初始{merge_hash}")
    while len(vocab) < vocab_size-len(special_tokens):
        if not merge_hash:
            break
        print(f"\r合并进度: {len(vocab)-256}/{vocab_size-256-len(special_tokens)}", end="")
        #!!
        #pair = max(merge_hash.items(), key=lambda x: (x[1], x[0]))[0]
        # 先找到最大频率
        # max_freq = max(merge_hash.values())

        # # 找出所有频率等于最大值的 pair
        # ties = [(pair, freq) for pair, freq in merge_hash.items() if freq == max_freq]

        # if len(ties) > 1:
        #     print("⚠️ 出现频率相同的 pair：")
        #     for pair, freq in ties:
        #         print(f"  pair=({vocab[pair[0]]},{vocab[pair[1]]!r}), freq={freq}")

        #然后再按原来的规则选出字典序最大的
        pair = max(
            merge_hash.items(),
            key=lambda x: (
                x[1],  # 频率优先
                vocab[x[0][0]],
                vocab[x[0][1]]
            )
        )[0]
        # if len(ties) > 1:
        #     print(f"  pair=({vocab[pair[0]]},{vocab[pair[1]]!r}), freq={freq}")
        index1, index2 = pair
        merges.append((vocab[index1],vocab[index2]))
        vocab[next_idx] = vocab[index1] + vocab[index2]
        #分发任务
        for _ in range(num_processes):
            task_queue.put((pair,next_idx))
        
        merge_hash = Counter()
        for _ in range(num_processes):
            local_merge_hash = result_queue.get()
            if local_merge_hash is None:
                continue
            merge_hash.update(local_merge_hash)
        #print(f"主进程{merge_hash}")
        #忘了
        next_idx+=1
    #结束进程
    for _ in range(num_processes):
        task_queue.put(None)
    for p in processes:
        p.join()
    print("[主进程] 所有子进程已退出")
    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8") #得到utf-8编码，也是字节流
        next_idx += 1
    return vocab,merges

import json

def save_vocab(vocab, output_path: str):
    # 把 bytes 转成 utf-8 字符串
    vocab_str = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
        
import cProfile
import pstats

# merges: [(b'a', b'b'), (b'ab', b'c')]
def save_merges(merges, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            token1 = b1.decode("utf-8", errors="replace")
            token2 = b2.decode("utf-8", errors="replace")
            f.write(f"{token1} {token2}\n")



if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    vocab,merges = train_bpe(
        input_path= r"E:\cs336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt",
        vocab_size = 1000,
        special_tokens = ["<|endoftext|>"]   
    )

    save_vocab(vocab, "vocab1.json") 
    save_merges(merges, "merges1.txt")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")  # 按累计耗时排序
    stats.print_stats(20)  # 打印前 20 个最耗时的函数
