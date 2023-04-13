import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource


# Define the Flask app
app = Flask(__name__)
api = Api(app)

# Load the trained model
model = joblib.load('./trained_model.pkl')



class Recommend(Resource):
    def post(self):

        target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']


        # Get the request data
        request_data = request.get_json()

        # Extract the input features
        ncodpers = request_data['ncodpers']
        renta = request_data['renta']
        age = request_data['age']
        ind_nuevo = request_data['ind_nuevo']
        antiguedad = request_data['antiguedad']
        indrel_1mes = request_data['indrel_1mes']
        tiprel_1mes = request_data['tiprel_1mes']
        indresi = request_data['indresi']
        indext = request_data['indext']
        conyuemp = request_data['conyuemp']
        canal_entrada = request_data['canal_entrada']

        df = pd.DataFrame({
            'ncodpers': [ncodpers],
            'renta': [renta],
            'age': [age],
            'ind_nuevo': [ind_nuevo],
            'antiguedad': [antiguedad],
            'indrel_1mes': [indrel_1mes],
            'tiprel_1mes': [tiprel_1mes],
            'indresi': [indresi],
            'indext': [indext],
            'conyuemp': [conyuemp],
            'canal_entrada': [canal_entrada]
        })

        df['ncodpers'] = df['ncodpers'].astype(int)
        df['renta'] = df['renta'].astype(float)
        df['age'] = df['age'].astype(int)
        df['ind_nuevo'] = df['ind_nuevo'].astype(int)
        df['antiguedad'] = df['antiguedad'].astype(int)
        df['indrel_1mes'] = df['indrel_1mes'].astype(str).astype(int)
        df['tiprel_1mes'] = df['tiprel_1mes'].astype('category')
        df['indresi'] = df['indresi'].astype('category')
        df['indext'] = df['indext'].astype('category')
        df['conyuemp'] = df['conyuemp'].astype('category')
        df['canal_entrada'] = df['canal_entrada'].astype('category')

        # Convert the DataFrame to a DMatrix object
        dmatrix = xgb.DMatrix(df, enable_categorical=True)

        # Make predictions
        preds = model.predict(dmatrix)

        # getting top products
        target_cols = np.array(target_cols)

        preds = np.argsort(preds, axis=1)
        preds = np.fliplr(preds)[:,:7]

        final_preds = [", ".join(list(target_cols[pred])) for pred in preds]

        # Return the recommended products as a JSON response
        return jsonify({'recommended_products': final_preds})


api.add_resource(Recommend, '/recommend')

if __name__=="__main__":
    app.run(host='0.0.0.0')