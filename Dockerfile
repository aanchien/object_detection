# Use an official Python runtime as a parent image
FROM python:3

# Set the working directory to /app
WORKDIR /object_detection

# Copy the current directory contents into the container at /app
COPY requirements.txt  /object_detection/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt 

# Copy the current directory contents into the container at /app
COPY . /object_detection

# Run detect.py when the container launches
CMD ["python", "detect.py"]