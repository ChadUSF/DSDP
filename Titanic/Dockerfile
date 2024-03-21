# Use an official Jupyter Notebook data science image
FROM jupyter/datascience-notebook:latest

# Copy the notebook files into the container
COPY . /home/jovyan/work

RUN pip install tensorflow

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''"]