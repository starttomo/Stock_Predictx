# database/models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from decimal import Decimal

db = SQLAlchemy()

# database/models.py
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    cash = db.Column(db.DECIMAL(15, 4), default=1000000.0)  # 增加小数位数
    total_invest = db.Column(db.DECIMAL(15, 4), default=100000.0)
    current_value = db.Column(db.DECIMAL(15, 4), default=100000.0)
    profit = db.Column(db.DECIMAL(15, 4), default=0.0)
    profit_rate = db.Column(db.DECIMAL(10, 4), default=0.0)  # 修正为10,4
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

class Trade(db.Model):
    __tablename__ = 'trades'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True)
    buy_date = db.Column(db.Date, nullable=False)
    buy_price = db.Column(db.DECIMAL(10, 2), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    sell_date = db.Column(db.Date, nullable=True)
    sell_price = db.Column(db.DECIMAL(10, 2), nullable=True)
    order_type = db.Column(db.Enum('market', 'limit'), default='market')
    limit_price = db.Column(db.DECIMAL(10, 2), nullable=True)
    stop_loss = db.Column(db.DECIMAL(10, 2), nullable=True)
    take_profit = db.Column(db.DECIMAL(10, 2), nullable=True)
    status = db.Column(db.Enum('pending', 'filled', 'cancelled'), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Trade {self.quantity}股 @ {self.buy_price}>'

class Forecast(db.Model):
    __tablename__ = 'forecasts'
    date = db.Column(db.Date, primary_key=True)
    yhat = db.Column(db.Float)
    yhat_lower = db.Column(db.Float)
    yhat_upper = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Forecast {self.date}: {self.yhat}>'