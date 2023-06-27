# Base image from PyTorch
FROM pytorch/pytorch

# Set up for your local zone and UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory in the container
WORKDIR /app

# Copy the required files into the container
COPY requirements.txt .
COPY model.py .
COPY main.py .
COPY recommender_app.py .
COPY model_svm.pkl .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run when the container starts
CMD ["python", "recommender_app.py", "--model", "model_svm.pkl"]

