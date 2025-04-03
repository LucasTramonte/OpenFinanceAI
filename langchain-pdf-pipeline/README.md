# LangChain PDF Processing Pipeline

This project implements a LangChain pipeline for processing PDF documents and generating responses based on user queries. The pipeline consists of several modules that work together to convert PDFs into images, generate embeddings, handle queries, and produce responses.

## Project Structure

```
langchain-pdf-pipeline
├── src
│   ├── main.py                # Entry point for the LangChain pipeline
│   ├── pipeline
│   │   ├── __init__.py        # Initializes the pipeline package
│   │   ├── pdf_processing.py   # Functions for PDF to image conversion
│   │   ├── embeddings.py       # Functions for generating embeddings
│   │   ├── query_handling.py   # Functions for handling user queries
│   │   └── response_generation.py # Functions for generating responses
│   └── utils
│       ├── __init__.py        # Initializes the utils package
│       ├── logging.py          # Logging configuration
│       └── file_operations.py   # Utility functions for file operations
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd langchain-pdf-pipeline
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the pipeline, execute the following command:

```
python src/main.py
```

Make sure to update the `main.py` file with the appropriate PDF path and query before running.

## Functionality

- **PDF Processing**: Converts PDF documents into images for further processing.
- **Embedding Generation**: Generates embeddings from the images using a pre-trained model.
- **Query Handling**: Processes user queries to retrieve relevant pages based on similarity scoring.
- **Response Generation**: Generates responses based on the top-k relevant pages and saves them to a file.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.