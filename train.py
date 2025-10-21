#from cs336_basics.pretokenization_example import find_chunk_boundaries,find_chunk_boundaries_debug
from cs336_basics.pretokenization_example import find_chunk_boundaries,find_chunk_boundaries_debug
import regex as re
from collections import defaultdict
#from cs336_basics.BPE import BPETokenizer,BPETokenizerParams,get_compression_ratio
from cs336_basics.BPE import BPETokenizer,BPETokenizerParams,get_compression_ratio
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import heapq

import json
from tests.common import gpt2_bytes_to_unicode
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
class Node:
    """表示词内一个 token 节点，便于链表原地更新。"""
    def __init__(self, value, word_freq):
        self.value = value
        self.word_freq = word_freq  # 共享引用，节省内存
        self.prev = None
        self.next = None

class PQItem:
    """定义优先队列元素，实现自定义比较：频率优先，其次按字典序逆序。"""
    def __init__(self, freq, id_pair, byte_pair):
        self.freq = freq
        self.id_pair = id_pair
        self.byte_pair = byte_pair

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq  # 频率高的先出
        return self.byte_pair > other.byte_pair  # 字典序大的先出

def pre_tokenize_and_count(
    args: tuple[bytes, list[str]]
) -> Counter:
    chunk_bytes, special_tokens = args
    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        #!!
        #chunks = re.split(pattern, all_chunks)#去掉了special_token的大chunk
        chunk = [c for c in re.split(pattern, chunk) if c]
        #为什么一定要去掉空的：
    else:
        chunk = [chunk] #再按照special_tokens切分为小段

    words_list = [] #最后统一转化为Counter
    for s_chunk in chunk:
        for match in re.finditer(PAT, s_chunk):
            byte_sequence = match.group(0).encode("utf-8")
            id_sequence = tuple(byte_sequence)
            words_list.append(id_sequence)
    return Counter(words_list)

def train_bpe(
        input_path:str,
        vocab_size: int,
        special_tokens: list[str] = None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_idx = 256
    merges:list[tuple[int, int]] = []
    special_tokens = special_tokens or []

    #准备多进程参数
    num_processes = 4
    desired_num_chunks = 4
    with open(input_path, "rb") as f:
        desired_num_chunks = 4
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_args.append(
                (
                    chunk_bytes,
                    special_tokens
                )
            )
    
    processes_to_use = num_processes
    if processes_to_use is None:
        processes_to_use = min(cpu_count(), 8)

    processes_to_use = min(processes_to_use, len(chunk_args))
    all_word_freqs = Counter()#总的词频统计 turpe(int,)->int
    start_time = time.time()

    with Pool(processes=processes_to_use) as pool:
        print(
            f"Starting pre-tokenization with {processes_to_use} processes on {len(chunk_args)} chunks..."
        )
        results_iterator = pool.imap_unordered(pre_tokenize_and_count, chunk_args)
        for chunk_counter in tqdm(
            results_iterator, total=len(chunk_args), desc="Processing chunks", leave=True
        ):
            all_word_freqs.update(chunk_counter)#更新

    print(f"Pre-tokenization and initial counting time: {time.time() - start_time:.2f} seconds")
    
    ### Pre-tokenization 结束
    # 得到链表和pair到node的映射(pair指向链表)
    pair_to_nodes = defaultdict(set)
    for word_tuple, count in tqdm(all_word_freqs.items(), desc="Building", leave=True):
        if len(word_tuple) < 2:
            continue
        # 所有链表节点共享 word_freq 引用，节省内存
        word_freq = {'count': count}
        head =  Node(word_tuple[0], word_freq)
        prev_node = head
        for i in range(1,len(word_tuple)):
            curr_node = Node(word_tuple[i], word_freq)
            prev_node.next = curr_node  
            curr_node.prev = prev_node 

            pair = (prev_node.value, curr_node.value) #int和word_freq
            pair_to_nodes[pair].add(prev_node)#只加左节点
            prev_node = curr_node  
    
    del all_word_freqs

    #统计pair次数并建立最大堆
    pair_freqs = Counter()
    for pair, nodes in tqdm(pair_to_nodes.items(), desc="Counting pairs", leave=True):
        pair_freqs[pair] = sum(node.word_freq["count"] for node in nodes)
    pq = [
        PQItem(freq, p, (vocab[p[0]], vocab[p[1]]))
        for p,freq in pair_freqs.items()
    ]
    heapq.heapify(pq)

    ### BPE 开始
    merges = []
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    pbar = tqdm(total=num_merges, desc="Performing BPE merges")

    start_time = time.time()
    for _ in range(num_merges):
        if not pq:
            break

         # 取出频率最高的 pair，处理优先队列惰性删除的过期元素
        best_pair = None
        while pq:#找到对应的pair再继续执行，如果没有说明结束了
            item = heapq.heappop(pq)
            if item.id_pair not in pair_freqs:
                continue  # 已经被合并删除
            if pair_freqs[item.id_pair] == item.freq:
                best_pair = item.id_pair
                break

        if best_pair is None:
            break
        
        p1, p2 = best_pair

        # 合成新 token，添加到 merges/vocab
        merged_token_bytes = vocab[p1] + vocab[p2]
        merges.append((vocab[p1], vocab[p2]))
        vocab[next_idx] = merged_token_bytes


        # 逐个更新包含改 pair 的词
        nodes_to_process = list(pair_to_nodes[best_pair])
        for node1 in nodes_to_process:
            node2 = node1.next
            if node2 is None:
                continue#已经被删掉了（合并了）
            word_freq = node1.word_freq['count']
            if node1.prev:#需要更新链表，pair_to_nodes,pair_freqs,以及堆
                left = node1.prev

                old_left_pair = (left.value, node1.value)
                pair_freqs[old_left_pair] -= word_freq
                #修改后的pair_freqs
                heapq.heappush(pq, PQItem(pair_freqs[old_left_pair], old_left_pair, (vocab[old_left_pair[0]], vocab[old_left_pair[1]])))

                pair_to_nodes[old_left_pair].discard(left)#set add discard

                new_left_pair = (left.value, next_idx)
                pair_to_nodes[new_left_pair].add(left)
                pair_freqs[new_left_pair] += word_freq
                heapq.heappush(pq, PQItem(pair_freqs[new_left_pair], new_left_pair, (vocab[new_left_pair[0]], vocab[new_left_pair[1]])))
            
            if node2.next:
                right = node2.next
                old_right_pair = (node2.value, right.value)
                pair_freqs[old_right_pair] -= word_freq
                heapq.heappush(pq, PQItem(pair_freqs[old_right_pair], old_right_pair, (vocab[old_right_pair[0]], vocab[old_right_pair[1]])))

                new_right_pair = (next_idx, right.value)
                pair_to_nodes[old_right_pair].discard(node2)
                pair_to_nodes[new_right_pair].add(node1)
                pair_freqs[new_right_pair] += word_freq
                heapq.heappush(pq, PQItem(pair_freqs[new_right_pair], new_right_pair, (vocab[new_right_pair[0]], vocab[new_right_pair[1]])))
            
            # 链表合并：node1、node2合成 next_idx
            node1.value = next_idx #更新node1
            node1.next = node2.next #丢掉node2
            if node2.next:
                node2.next.prev = node1

        del pair_freqs[best_pair]
        del pair_to_nodes[best_pair]
        next_idx += 1
        pbar.update(1)
    
    end_time = time.time()
    print(f"Merge time: {end_time - start_time:.2f} seconds")
    pbar.close()
    
    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8") #得到utf-8编码，也是字节流
        next_idx += 1
    return vocab,merges

def save_vocab(vocab, output_path: str):
    byte_encoder = gpt2_bytes_to_unicode()
    vocab_str = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)
        


# merges: [(b'a', b'b'), (b'ab', b'c')]
def save_merges(merges, output_path: str):
    """
    merges: list[tuple[bytes, bytes]]
    保存为文本文件，每行 "token1 token2"
    token 内部字节用 GPT-2 字节编码转成可见字符
    """
    byte_encoder = gpt2_bytes_to_unicode()
    with open(output_path, "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            token1 = "".join(byte_encoder[b] for b in b1)
            token2 = "".join(byte_encoder[b] for b in b2)
            f.write(f"{token1} {token2}\n")
import cProfile
import pstats
import pathlib
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    BASE_DIR = pathlib.Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "../data/TinyStoriesV2-GPT4-valid.txt"

    vocab,merges = train_bpe(
        input_path= DATA_PATH,
        vocab_size = 10000,
        special_tokens = ["<|endoftext|>"]   
    )

    save_vocab(vocab, "vocab.json") 
    save_merges(merges, "merges.txt")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")  # 按累计耗时排序
    stats.print_stats(20)  # 打印前 20 个最耗时的函数
