import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import torch
from transformers import BertTokenizer, BertModel

def read_file(file_path):
    """파일 경로에서 파일을 읽어와 내용을 반환"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def calculate_cosine_similarity_bert(embedding1, embedding2):
    """BERT 임베딩을 사용한 코사인 유사도 계산"""
    cosine_sim = cosine_similarity(embedding1, embedding2)
    return cosine_sim[0][0]

def calculate_structural_similarity(text1, text2):
    """코드의 구조적 유사성을 계산 (간단한 LCS 사용)"""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()

def get_r_files_in_directory(directory):
    """주어진 디렉토리 내의 모든 .r 파일을 찾음"""
    r_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.r')]
    return r_files

def get_bert_embeddings(text_list):
    """BERT를 사용하여 텍스트 리스트의 임베딩을 계산 + CUDA"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()




    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # 텍스트를 토큰화하고 패딩 및 어텐션 마스크를 만듭니다.
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # BERT의 [CLS] 토큰에 대한 임베딩을 추출
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    """
    return embeddings

def compare_files(file_list):
    """파일 리스트 내의 모든 파일을 서로 비교하여 유사도를 계산"""
    similarity_results = []

    # 모든 파일 내용 읽기
    texts = [read_file(file) for file in file_list]

    # BERT 임베딩 계산
    embeddings = get_bert_embeddings(texts)

    for (i, file1), (j, file2) in combinations(enumerate(file_list), 2):
        # BERT 임베딩 기반 유사도 계산
        cosine_sim_bert = calculate_cosine_similarity_bert([embeddings[i]], [embeddings[j]])

        # 구조적 유사도 계산
        structural_sim = calculate_structural_similarity(texts[i], texts[j])

        # 결과 저장
        similarity_results.append((file1, file2, cosine_sim_bert, structural_sim))

    return similarity_results

def main():
    # 사용자로부터 폴더 경로 입력받기
    directory = input("코드 파일들이 있는 폴더 경로를 입력하세요: ")

    # 폴더 유효성 검사
    if not os.path.isdir(directory):
        print(f"유효하지 않은 폴더 경로입니다: {directory}")
        return

    # 디렉토리 내의 모든 R 파일 찾기
    r_files = get_r_files_in_directory(directory)

    if len(r_files) < 2:
        print("비교할 R 파일이 2개 이상 있어야 합니다.")
        return

    # 파일들 간의 유사도 비교
    similarity_results = compare_files(r_files)

    # 유사도가 높은 순으로 정렬
    sorted_results = sorted(similarity_results, key=lambda x: (x[2], x[3]), reverse=True)

    # 결과 출력
    print("\n유사도가 높은 파일 쌍들:")
    for file1, file2, cosine_sim_bert, structural_sim in sorted_results:
        print(f"파일: {os.path.basename(file1)} - {os.path.basename(file2)} | BERT 코사인 유사도: {cosine_sim_bert:.2f} | 구조적 유사도: {structural_sim:.2f}")

if __name__ == "__main__":
    main()