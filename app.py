from flask import Flask, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import chardet
import matplotlib
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
import functools
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import json
import getpass

