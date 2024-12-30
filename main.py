from data_loader import load_data_from_csv
from ml_model import predictor
from models import PassengerModel, PassengerData

from fastapi import FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select



DATABASE_URL = "sqlite+aiosqlite:///./titanic.db"
engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)

app = FastAPI()
Base = declarative_base()


@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as session:
        csv_file_path = './Data_for_Titanic/survival_probabilities.csv'
        await load_data_from_csv(session, csv_file_path)

# endpoint для проверки выживет ли пассажир
@app.post("/predict_survival/")
async def predict_survival(passenger_data: PassengerData):
    passenger_dict = passenger_data.dict()
    probability = predictor(passenger_data=passenger_dict)
    return {"survival_probability": probability}

# endpoint для запроса данных о пассажире
@app.get("/passenger/{passenger_id}")
async def get_passenger(passenger_id: int):
    async with SessionLocal() as session:
        result = await session.execute(select(PassengerModel).where(PassengerModel.id == passenger_id))
        passenger = result.scalars().first()
        if passenger is None:
            raise HTTPException(status_code=404, detail="Passenger not found")
        return {
            "id": passenger.id,
            "Pclass": passenger.Pclass,
            "Sex": passenger.Sex,
            "Age": passenger.Age,
            "SibSp": passenger.SibSp,
            "Parch": passenger.Parch,
            "Fare": passenger.Fare,
            "Embarked": passenger.Embarked
        }

# endpoint для добавления пассажира
@app.post("/add_passenger/")
async def add_passenger(passenger_data: PassengerData):
    async with SessionLocal() as session:
        new_passenger = PassengerModel(**passenger_data.dict())
        session.add(new_passenger)
        await session.commit()
        return {"message": "Passenger added successfully"}

# endpoint для удаления пассажира
@app.delete("/delete_passenger/{passenger_id}")
async def delete_passenger(passenger_id: int):
    async with SessionLocal() as session:
        result = await session.execute(select(PassengerModel).where(PassengerModel.id == passenger_id))
        passenger = result.scalars().first()
        if passenger is None:
            raise HTTPException(status_code=404, detail="Passenger not found")
        await session.delete(passenger)
        await session.commit()
        return {"message": "Passenger deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 