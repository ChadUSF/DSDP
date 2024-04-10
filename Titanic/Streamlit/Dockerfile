FROM python:3.12

# Set working directory
WORKDIR /app

# Copy files into the docker container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "titanicSL.py", "--server.port=8501", "--server.address=0.0.0.0"]