# goorm_ai_mission
2025년 하반기 구름 인턴십 채용(AI엔지니어) 미션

## 프로젝트 개요
해당 프로젝트는 위키피디아 데이터(KorQuAD)셋을 활용하여 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 것으로 REST API 형태로 질의응답 서비스를 제공하는 것이 목표이다.<br/>

### 기술스택
**언어**: Python<br/>
**활용 툴**: HuggingFace, FAISS, FastAPI<br/>
**활용 모델**: snowflake, Qwen3-4B<br/>

### 설치 및 실행 방법

설치 및 실행 방법은 아래와 같다.<br/>
Google Colab으로 실행할 경우 [해당 예시](https://colab.research.google.com/drive/1_8To3qdZjZ6hJviuvJ6o8XbfAkyqotzY?usp=sharing)를 통해 실행 가능하다. **(단, pyngrok 토큰 필요)**

#### 설치

retrieve에 필요한 벡터DB는 github에 올려 놓았으므로 `gitclone`만으로 가능하다.<br/>

1. gitclone
    ```
    git clone https://github.com/thplus/goorm_ai_mission.git
    ```

2. `requirements` 설치

    ```
    cd goorm_ai_mission
    ```

    ```
    pip install -r requirements.txt
    ```

#### 실행

1. fastapi 실행
    ```
    cd goorm_ai_mission/app
    ```

    ```
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

2. /ask 요청 <br/>

    `127.0.0.1:8000` 기준 아래와 같다.
    
    ```
    curl -X 'POST' \
    'http://127.0.0.1:8000/ask' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "question": "바그너는 교향곡 작곡을 어디까지 쓴 뒤에 중단했는가?"
    }'
    ```

### API 사용 예시

요청
```json
{
  "question": "바그너는 교향곡 작곡을 어디까지 쓴 뒤에 중단했는가?"
}
```

|이름|타입|설명|
|--|--|--|
|`question`|string|질문할 내용|

응답
```json
{
  "retrieved_document_id": "13130",
  "retrieved_document": "그는 교향곡을 2곡 썼는데, 1834년에 기회음악으로 작곡한 교향곡 마장조는 제1악장과 아다지오 2악장의 29마디까지만 쓰고 미완성으로 끝났다. (나중에 2악장은 보필됨.) 따라서 바그너가 완성한 교향곡으로는 이 교향곡 다장조를 들 수 있다. 이 교향곡은 형식과 작곡기법면에서 놀랄 만한 완성도를 가지고 있다. 이 작품이 크리스티안 바인리히(Christian Theodor Weinlig, 1780~1842, 클라라 슈만의 작곡 스승으로도 알려져 있음, 토마스 교회의 지휘자) 밑에서 처음으로 작곡 수업을 마치고 난 직후인 19세 청년의 작품인 것을 감안할 때, 그 힘있는 음악 기법은 가히 감탄할 만한 것이다. 확고한 형식이 유지되고, 정돈된 음악적 대비를 멋지게 담아내었을 뿐 아니라 정교한 조바꿈도 자주 나타나므로 풍부한 울림을 감상할 수 있다. 바그너는 1830년경부터 많은 연주회용 서곡을 작곡하였으며 그런 경험이 교향곡 작곡에 자신감을 갖게 했을 것이다. 이 곡은 1832년 6월에 라이프치히에서 작곡하였다.",
  "question": "바그너는 교향곡 작곡을 어디까지 쓴 뒤에 중단했는가?",
  "answers": "바그너는 교향곡 작곡을 1악장을 쓴 뒤에 중단했다."
}
```

|이름|타입|설명|
|--|--|--|
|`retrieved_document_id`|int|내부 row_id|
|`retrieved_document`|string|LLM이 실제 참고한 문서 내용|
|`question`|string|질문한 내용|
|`answers`|string|답변|

## 구현 아키텍처 및 접근 방식 설명
[Project Description](/project_description.md) 참고
