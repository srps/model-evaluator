
# Embedding Model and Search Strategy Evaluator

This project provides a tool for evaluating various embedding models and search strategies using a Streamlit web application. It allows users to upload a corpus, select multiple embedding models, and compare different search strategies for information retrieval tasks.

## Features

- Upload custom corpus (TXT file)
- Support for multiple embedding models:
  - TensorFlow Serving models (via TensorFlow Serving API)
  - Sentence Transformer models (via Sentence Transformers library)
- Evaluate various search strategies:
  - Exact Match
  - Prefix Match
  - Fuzzy Match
  - BM25 Retrieval
  - Semantic Search (using selected embedding models)
- Interactive weight adjustment for each search strategy
- Real-time visualization of search results and execution times
- On-demand downloading of additional Sentence Transformer models

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Application

1. Clone this repository
2. Navigate to the project directory
3. Run the following command:

```bash
docker compose up --build
```

4. Open your browser and go to http://localhost:8502 to access the Streamlit application

## Developing with Docker Watch

To develop while running the application, you can run instead:
```bash
docker compose watch
```

### Project Structure

* docker-compose.yml: Defines the services for TensorFlow Serving and the Streamlit application
* Dockerfile.streamlit: Dockerfile for building the Streamlit application image
* src/app.py: Main Streamlit application code
* models/: Directory for storing downloaded Sentence Transformer models

### Usage

1. Upload a corpus (TXT file) using the file uploader
2. Select the embedding models you want to evaluate
3. Enter a search query
4. Choose the search strategies to use and adjust their weights
5. View the search results, including scores for each strategy and overall weighted scores
6. Analyze the execution time chart for performance comparison

### Customization

* To add new TensorFlow Serving models, update the `tf_serving` service in `docker-compose.yml`
* To use different Sentence Transformer models, download them using the provided interface in the Streamlit app

## Attributions

* [Streamlit](https://streamlit.io/)
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
* [Sentence Transformers](https://www.sbert.net/)
* [BM25 Retrieval](https://github.com/xhluca/bm25s)
* [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

## Contributing

This is a personal project that helped me quickly evaluating embedding models and search strategies. If you find it useful, feel free to use for your own purposes.
While contributions are welcome, please note that this project is just a scrappy tool and not a production-ready solution.

## License
This project is open-source and available under the MIT License.

