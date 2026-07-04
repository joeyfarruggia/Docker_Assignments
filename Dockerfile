#Use a base image that has the necessary dependencies for your script
FROM python:3.9

#Set the working directory in the container
WORKDIR /app

#Copy the AI Code into the container
COPY main.py .

#Install any required dependencies (if any)
RUN pip install numpy tensorflow

#Set the command to run your AI Code
CMD ["python", "main.py"]