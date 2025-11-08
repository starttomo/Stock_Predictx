from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from decimal import Decimal

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    total_invest = db.Column(db.Numeric(10, 2), default=Decimal('1000000.00'))
    cash = db.Column(db.Numeric(10, 2), default=Decimal('1000000.00'))
    current_value = db.Column(db.Numeric(10, 2), default=Decimal('1000000.00'))
    profit = db.Column(db.Numeric(10, 2), default=Decimal('0.00'))
    profit_rate = db.Column(db.Numeric(5, 2), default=Decimal('0.00'))
    holding_quantity = db.Column(db.Integer, default=0)  # 添加持仓量字段

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    buy_date = db.Column(db.DateTime, nullable=False)
    buy_price = db.Column(db.Numeric(10, 2), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    sell_date = db.Column(db.DateTime)
    sell_price = db.Column(db.Numeric(10, 2))
    profit = db.Column(db.Numeric(10, 2))
    profit_rate = db.Column(db.Numeric(5, 2))
    status = db.Column(db.String(20), default='pending')
    order_type = db.Column(db.String(20), default='market')
    side = db.Column(db.String(10), default='buy')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Forecast(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    yhat = db.Column(db.Numeric(10, 2), nullable=False)
    yhat_lower = db.Column(db.Numeric(10, 2), nullable=False)
    yhat_upper = db.Column(db.Numeric(10, 2), nullable=False)