from pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict

# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
file_path="../data/TinyStoriesV2-GPT4-valid.txt"
file_path="../data/test.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def worker_wrong(
        input_path:str,
        vocab_size: int,  
        start:int,
        end:int,
        special_tokens: list[str] = None
):
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_idx = 256
    merges: dict[tuple[int, int], int] = {}
    special_tokens = special_tokens or []

    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1
        #new_idx还会在merge中使用

    with open(input_path,"rb") as f:
        f.seek(start)
        data = f.read() if end is None else f.read(end - start)
        all_chunks = data.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, all_chunks)#去掉了special_token的大chunk
    else:
        chunks = [all_chunks]
    
    near_pair_dict: dict[tuple[int, int], tuple[defaultdict(int), defaultdict(int)]] = \
    defaultdict(lambda: (defaultdict(int), defaultdict(int)))#工厂函数提供默认值
    merge_hash = defaultdict(int)

    print("拆分结果:", chunks)
    
    first = True
    while len(vocab) < vocab_size:#到上线之前一直循环
        if(first):
            for chunk in chunks:#循环每个去掉了special_token的大chunk
                for match in re.finditer(PAT, chunk):#对每个chunk进行分词
                    pretoken = match.group(0)#分词后不再进行跨pretoken的统计
                    pretoken = list(map(int, pretoken.encode("utf-8")))
                    # merge_hash: dict[tuple[int, int],int]
                    # pretoken: list[int]
                    for i, (index1, index2) in enumerate(zip(pretoken, pretoken[1:])):
                        pair = (index1, index2)
                        merge_hash[pair] += 1
                        # 记录前后邻居
                        prev_dict, next_dict = near_pair_dict[pair]
                        if i > 0:
                            prev_pair = (pretoken[i-1], pretoken[i])
                            prev_dict[prev_pair]+=1
                        if i + 2 <= len(pretoken) - 1:
                            next_pair = (pretoken[i+1], pretoken[i+2])
                            next_dict[next_pair]+=1
            first = False

        pair = max(merge_hash, key=merge_hash.get)
        #max_value = merge_hash[pair]
        index1, index2 = pair
        #Merge that pair.
        #next_idx
        merges[pair] = next_idx  # @inspect merges
        vocab[next_idx] = vocab[index1] + vocab[index2]  # @inspect vocab
        
        #更新统计量
        prev_dict, next_dict = near_pair_dict[pair]
        for prev_pair, value in prev_dict.items():
            previdx, idx = prev_pair
            merge_hash[prev_pair]-=value
            merge_hash[(previdx , next_idx)]+=value
        for next_pair, value in next_dict.items():
            idx, nextidx = next_pair
            merge_hash[next_pair]-=value
            merge_hash[(next_idx , nextidx)]+=value

        merge_hash.pop(pair, None)
        near_pair_dict.pop(pair, None)

        next_idx += 1

    return vocab,merges

def worker(
        input_path:str,
        vocab_size: int,
        start:int,
        end:int,
        special_tokens: list[str] = None
):
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_idx = 256
    merges: dict[tuple[int, int], int] = {}
    special_tokens = special_tokens or []

    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1
        #new_idx还会在merge中使用

    with open(input_path,"rb") as f:
        f.seek(start)
        data = f.read() if end is None else f.read(end - start)
        all_chunks = data.decode("utf-8", errors="ignore")
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, all_chunks)#去掉了special_token的大chunk
    else:
        chunks = [all_chunks]
    
    
    # defaultdict(lambda: (defaultdict(int), defaultdict(int)))#工厂函数提供默认值
    merge_hash = defaultdict(int)
    #修改
    occ = defaultdict(list)  # pair -> [(chunk_id,pretoken_id,pos), ...]
    
    #print("拆分结果:", chunks)
    pretoken_list = []
    for chunk_id, chunk in enumerate(chunks):#循环每个去掉了special_token的大chunk
        pretokens  = []
        for pretoken_id,match in enumerate(re.finditer(PAT, chunk)):#对每个chunk进行分词
            pretoken = match.group(0)#分词后不再进行跨pretoken的统计
            pretoken = list(map(int, pretoken.encode("utf-8")))
            pretokens.append(pretoken)
            # merge_hash: dict[tuple[int, int],int]
            # pretoken: list[int]
            for i, (index1, index2) in enumerate(zip(pretoken, pretoken[1:])):
                pair = (index1, index2)
                merge_hash[pair] += 1
                occ[pair].append((chunk_id,pretoken_id,i))
        pretoken_list.append(pretokens)

    while len(vocab) < vocab_size:#到上线之前一直循环
        pair = max(merge_hash, key=merge_hash.get)
        #max_value = merge_hash[pair]
        index1, index2 = pair
        #Merge that pair.
        #next_idx
        merges[pair] = next_idx  # @inspect merges
        vocab[next_idx] = vocab[index1] + vocab[index2]  # @inspect vocab

        positions = sorted(occ[pair], key=lambda x: x[2], reverse=True)
        for chunk_id, pretoken_id, pos in positions:
            tokens = pretoken_list[chunk_id][pretoken_id]#加[:]就是副本
            if pos>0:#
                pre_idx = tokens[pos-1]
                if (pre_idx, index1) in merge_hash:
                    merge_hash[(pre_idx, index1)] -= 1
                    if merge_hash[(pre_idx, index1)] <= 0:
                        merge_hash.pop((pre_idx, index1))
                #occ[(pre_idx,index1)].remove((chunk_id,pretoken_id,pos-1))
                merge_hash[(pre_idx,next_idx)]+=1
                #occ[(pre_idx,next_idx)].append((chunk_id,pretoken_id,pos-1))
                
            if pos + 2 <= len(tokens)-1:
                nxt_idx = tokens[pos+2]#修改后会出现问题
                if (index2, nxt_idx) in merge_hash:
                    merge_hash[(index2, nxt_idx)]-=1
                    if merge_hash[(index2, nxt_idx)] <= 0:
                        merge_hash.pop((index2, nxt_idx))
                
                #occ[(index2,nxt_idx)].remove((chunk_id,pretoken_id,pos+1))
                merge_hash[(next_idx,nxt_idx)]+=1
                #occ[(next_idx,nxt_idx)].append((chunk_id,pretoken_id,pos))
            
            # 清理当前 pretoken 的 occ
            # 先清理原有token的occ
            for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
                if (chunk_id, pretoken_id, i) in occ[(a, b)]:
                    occ[(a, b)].remove((chunk_id, pretoken_id, i))

            tokens[pos:pos+2] = [next_idx] # 物理变动
            
            for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
                pair2 = (a, b)
                occ[pair2].append((chunk_id, pretoken_id, i))

        merge_hash.pop(pair, None)
        occ.pop(pair, None)     
        next_idx += 1

    return vocab,merges

from collections import Counter
 
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    if not pair:
        return indices

    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices      

def train_bpe(
        input_path:str,
        vocab_size: int,
        special_tokens: list[str] = None,
):
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    next_idx = 256
    merges: list[tuple[bytes, bytes]] = []
    special_tokens = special_tokens or []

    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1  # 给每个special token分配新的id
    start = 0
    with open(input_path,"rb") as f:
        f.seek(start)
        data = f.read()
        all_chunks = data.decode("utf-8", errors="ignore")    
        all_chunks = all_chunks.replace("\r\n", "\n").replace("\r", "\n")
    
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        #去掉空串
        chunks = [c for c in re.split(pattern, all_chunks) if c]
    else:
        chunks = [all_chunks]
    
    indices_per_token = [
        list(m.group(0).encode("utf-8"))  # 每个 pretoken 单独转字节序列
        for chunk in chunks
        for m in re.finditer(PAT, chunk)
    ]

    while len(vocab) < vocab_size:#到上线之前一直循环
        # merge_hash: dict[tuple[int, int],int]
        # pretoken: list[int]

        counts = Counter()
        for indices in indices_per_token:
            for a, b in zip(indices, indices[1:]):
                counts[(a, b)] += 1 
        
        pair = max(
        counts.items(),
        key=lambda x: (
            x[1],  # 频率优先
            vocab[x[0][0]],
            vocab[x[0][1]]
            )
        )[0]
        index1, index2 = pair
        # Merge that pair.
        # next_idx更新merges和vocab
        merges.append((vocab[index1],vocab[index2]))  # @inspect merges
        vocab[next_idx] = vocab[index1] + vocab[index2]  # @inspect vocab每次最多加一个 int->bytes
        
        #indices = merge(indices, pair, new_index)  # @inspect indices
        indices_per_token = [merge(indices, pair, next_idx) for indices in indices_per_token]
        next_idx = next_idx + 1

    return vocab,merges          


    