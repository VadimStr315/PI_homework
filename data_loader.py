import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import PassengerModel

async def load_data_from_csv(session: AsyncSession, csv_file_path: str):
    result = await session.execute(select(PassengerModel))
    passengers = result.scalars().all()

    if not passengers:  # If the database is empty
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            new_passenger = PassengerModel(
                Pclass=row['Pclass'],
                Sex=row['Sex'],
                Age=row['Age'],
                SibSp=row['SibSp'],
                Parch=row['Parch'],
                Fare=row['Fare'],
                Embarked=row['Embarked']
            )
            session.add(new_passenger)
        await session.commit()
        print("Data loaded from CSV into the database.")
    else:
        print("Database already contains data. No need to load from CSV.")