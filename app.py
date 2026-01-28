from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import BankData
from src.pipline.training_pipeline import Training_pipeline

app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from typing import Optional


class DataForm:
    """
    DataForm class to read and store bank marketing form data
    """

    def __init__(self, request: Request):
        self.request: Request = request

        # Numeric fields
        self.age: Optional[int] = None
        self.balance: Optional[int] = None
        self.day: Optional[int] = None
        self.duration: Optional[int] = None
        self.campaign: Optional[int] = None
        self.pdays: Optional[int] = None
        self.previous: Optional[int] = None

        # Categorical fields
        self.job: Optional[str] = None
        self.marital: Optional[str] = None
        self.education: Optional[str] = None
        self.default: Optional[str] = None
        self.housing: Optional[str] = None
        self.loan: Optional[str] = None
        self.contact: Optional[str] = None
        self.month: Optional[str] = None
        self.poutcome: Optional[str] = None

    async def get_form_data(self):
        """
        Reads form data from HTML form and converts it
        into correct Python data types
        """
        form = await self.request.form()

        # Convert numeric inputs
        self.age = int(form.get("age"))
        self.balance = int(form.get("balance"))
        self.day = int(form.get("day"))
        self.duration = int(form.get("duration"))
        self.campaign = int(form.get("campaign"))
        self.pdays = int(form.get("pdays"))
        self.previous = int(form.get("previous"))

        # Read categorical inputs
        self.job = form.get("job")
        self.marital = form.get("marital")
        self.education = form.get("education")
        self.default = form.get("default")
        self.housing = form.get("housing")
        self.loan = form.get("loan")
        self.contact = form.get("contact")
        self.month = form.get("month")
        self.poutcome = form.get("poutcome")

@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "form.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = Training_pipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.post("/")
async def predictRouteClient(request: Request):

    try:
        form = DataForm(request=request)
        await form.get_form_data() # this will fetch data from our form

        form_data = BankData(
            age = form.age,
            job = form.job,
            marital = form.marital,
            education = form.education,
            default=form.default,
            balance = form.balance,
            housing=form.housing,
            loan=form.loan,
            contact= form.contact,
            day=form.day,
            month= form.month,
            duration=form.duration,
            campaign = form.campaign,
            pdays = form.pdays,
            previous=form.previous,
            poutcome=form.poutcome
        )

        input_df = form_data.get_data_as_dataframe()

        prediction = form_data.predict(dataframe=input_df)[0]

        status = "Response-Yes" if prediction == 1 else "Response-No"

        return templates.TemplateResponse(
            "form.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)