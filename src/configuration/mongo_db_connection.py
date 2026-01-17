import os
import sys
import pymongo
import certifi

from src.exception import MyException
from src.logger import logger
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

ca = certifi.where()

class MongoDBClient:
    client = None


    def __init__(self,database_name:str=DATABASE_NAME):

        try:
            if MongoDBClient.client is None:
                mongodb_url = os.getenv(MONGODB_URL_KEY)
                if mongodb_url==None:
                    raise Exception(f"environmental variable {MONGODB_URL_KEY} is not set")
                MongoDBClient.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise MyException(e,sys)