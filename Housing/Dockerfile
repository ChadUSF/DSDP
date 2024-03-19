# Use an official Jupyter Notebook base image
FROM jupyter/base-notebook:latest

# Copy the notebook files into the container
COPY . /home/jovyan/work

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install mlxtend
RUN pip install scikit-learn

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''"]
