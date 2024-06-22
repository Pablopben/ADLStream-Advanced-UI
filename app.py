import ADLStream

import os
import json
from flask import Flask, render_template, request, send_file, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base

ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
engine = create_engine('sqlite:////tmp/adlstream.db')
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()
variables = {}
parameterValues = {}


def init_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(bind=engine)


class Results(Base):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    fileName = Column(String(200), unique=True)
    stream = Column(String(200))
    evaluator = Column(String(200))
    model = Column(String(200))

    def __init__(self, fileName=None, model=None, stream=None, evaluator=None):
        self.fileName = fileName
        self.model = model
        self.stream = stream
        self.evaluator = evaluator

    def __repr__self(self):
        return '<fileName>'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


init_db()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect('/stream')
    return render_template('home.html')


@app.route('/guide')
def guide():
    return render_template('guide.html')


@app.route('/stream', methods=['GET', 'POST'])
def stream():
    if request.method == 'POST':
        streamName = request.form['stream']
        variables.update({"streamName": streamName})
        if streamName == "CSVFileStream":
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                variables.update({"datasetName": filename[0:-4]})
                file.save("csv/" + filename)
                sep = request.form['sep']
                indexCol = int(request.form['indexCol'])
                header = int(request.form['header'])
                streamPeriod = int(request.form['streamPeriod'])
                timeout = int(request.form['timeout'])
                stream = ADLStream.data.stream.CSVFileStream("csv/" + filename,
                                                             sep=sep, header=header, stream_period=streamPeriod,
                                                             index_col=indexCol, timeout=timeout)
                streamParameters = {"CSV File Stream": {
                    "sep": sep,
                    "indexCol": indexCol,
                    "header": header,
                    "streamPeriod": streamPeriod,
                    "timeout": timeout
                }}
                parameterValues.update(streamParameters)
                variables.update({"stream": stream})
                return redirect("/selectGenerator")
        else:
            numFeatures = int(request.form['numFeatures'])
            streamLength = int(request.form['streamLength'])
            streamPeriod = int(request.form['streamPeriod'])
            stream = ADLStream.data.stream.FakeStream(
                num_features=numFeatures, stream_length=streamLength, stream_period=streamPeriod
            )
            streamParameters = {"Fake Stream": {
                "numFeatures": numFeatures,
                "streamLength": streamLength,
                "streamPeriod": streamPeriod
            }}
            parameterValues.update(streamParameters)
            variables.update({"stream": stream})
            return redirect("/selectGenerator")
    else:
        return render_template('streams/selectStream.html')


@app.route('/result')
def result():
    return render_template('result.html')


# @app.route('/streamForm', methods=['GET', 'POST'])
# def streamForm():
#     streamName = variables["streamName"]
#     if streamName == "CSVFileStream":
#         if request.method == 'GET':
#             return render_template('streams/csvStreamForm.html')
#         else:
#             if 'file' not in request.files:
#                 flash('No file part')
#                 return redirect(request.url)
#             file = request.files['file']
#             if file.filename == '':
#                 flash('No selected file')
#                 return redirect(request.url)
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 variables.update({"datasetName": filename[0:-4]})
#                 file.save("csv/" + filename)
#                 sep = request.form['sep']
#                 indexCol = int(request.form['indexCol'])
#                 header = int(request.form['header'])
#                 streamPeriod = int(request.form['streamPeriod'])
#                 timeout = int(request.form['timeout'])
#                 stream = ADLStream.data.stream.CSVFileStream("csv/" + filename,
#                                                              sep=sep, header=header, stream_period=streamPeriod,
#                                                              index_col=indexCol, timeout=timeout)
#                 variables.update({"stream": stream})
#                 return redirect("/selectGenerator")
#     else:
#         if request.method == 'GET':
#             return render_template('streams/fakeStreamForm.html', variables=variables)
#         else:
#             numFeatures = int(request.form['numFeatures'])
#             streamLength = int(request.form['streamLength'])
#             streamPeriod = int(request.form['streamPeriod'])
#             stream = ADLStream.data.stream.FakeStream(
#                 num_features=numFeatures, stream_length=streamLength, stream_period=streamPeriod
#             )
#             variables.update({"stream": stream})
#             return redirect("/selectGenerator")


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route('/selectGenerator', methods=['GET', 'POST'])
def selectGenerator():
    if request.method == 'POST':
        generator = request.form['generator']
        if generator == "classificationStreamGenerator":
            stream = variables["stream"]
            labelIndex = [int(x) for x in request.form["labelIndex"].split(",") if x != ""]
            oneHotLabels = [x.split() for x in request.form["oneHotLabels"].split(",") if x != ""]
            streamGenerator = ADLStream.data.ClassificationStreamGenerator(
                stream=stream, label_index=labelIndex, one_hot_labels=oneHotLabels
            )
            generatorParameters = {"classificationStreamGenerator": {
                "labelIndex": labelIndex,
                "oneHotLabels": oneHotLabels,
            }}
            parameterValues.update(generatorParameters)
            variables.update({"generator": streamGenerator})
        else:
            variables.update({"generatorName": generator})
            pastHistory = int(request.form['pastHistory'])
            forecastingHorizon = int(request.form['forecastingHorizon'])
            variables.update({"forecastingHorizon": forecastingHorizon})
            shift = int(request.form['shift'])
            stream = variables["stream"]
            streamGenerator = ADLStream.data.MovingWindowStreamGenerator(
                stream=stream, past_history=pastHistory, forecasting_horizon=forecastingHorizon, shift=shift
            )
            generatorParameters = {"movingWindowStreamGenerator": {
                "pastHistory": pastHistory,
                "forecastingHorizon": forecastingHorizon,
                "shift": shift
            }}
            parameterValues.update(generatorParameters)
            variables.update({"generator": streamGenerator})
        return redirect("/selectEvaluator")
    else:
        if variables["streamName"] == "CSVFileStream":
            return render_template('streamGenerators/classificationStreamGeneratorForm.html')
        else:
            return render_template('streamGenerators/movingWindowGeneratorForm.html')


# @app.route('/generatorForm', methods=['GET', 'POST'])
# def generatorForm():
#     generatorName = variables["generatorName"]
#     if generatorName == "classificationStreamGenerator":
#         if request.method == 'GET':
#             return render_template('streamGenerators/classificationStreamGeneratorForm.html')
#         else:
#             stream = variables["stream"]
#             labelIndex = [int(x) for x in request.form["labelIndex"].split(",") if x != ""]
#             oneHotLabels = [x.split() for x in request.form["oneHotLabels"].split(",") if x != ""]
#             streamGenerator = ADLStream.data.ClassificationStreamGenerator(
#                 stream=stream, label_index=labelIndex, one_hot_labels=oneHotLabels
#             )
#             variables.update({"generator": streamGenerator})
#     else:
#         if request.method == 'GET':
#             return render_template('streamGenerators/movingWindowGeneratorForm.html')
#         else:
#             pastHistory = int(request.form['pastHistory'])
#             forecastingHorizon = int(request.form['forecastingHorizon'])
#             variables.update({"forecastingHorizon": forecastingHorizon})
#             shift = int(request.form['shift'])
#             stream = variables["stream"]
#             streamGenerator = ADLStream.data.MovingWindowStreamGenerator(
#                 stream=stream, past_history=pastHistory, forecasting_horizon=forecastingHorizon, shift=shift
#             )
#             variables.update({"generator": streamGenerator})
#             return redirect("/selectEvaluator")


@app.route('/selectEvaluator', methods=['GET', 'POST'])
def selectEvaluator():
    if request.method == 'POST':
        evaluator = request.form['evaluator']
        variables.update({"evaluatorType": evaluator})
        if evaluator == "interleavedChunkEvaluator":
            chunkSize = int(request.form['chunkSize'])
            metric = request.form['metric']
            datasetName = request.form['datasetName']
            datasetName = checkDatasetName(datasetName, 0)
            evaluator = ADLStream.evaluation.InterleavedChunkEvaluator(
                chunk_size=chunkSize,
                metric=metric,
                results_file="static/results/" + datasetName + " Results.csv",
                dataset_name=datasetName,
                show_plot=True,
                plot_file="static/test.jpg"

            )
            evaluatorParameters = {"interleavedChunkEvaluator": {
                "chunkSize": chunkSize,
                "metric": metric
            }}
            parameterValues.update(evaluatorParameters)
            variables.update({"evaluator": evaluator, "datasetName": datasetName})
            return redirect("/selectModel")
        else:
            chunkSize = int(request.form['chunkSize'])
            faddingFactor = float(request.form['faddingFactor'])
            metric = request.form['metric']
            datasetName = request.form['datasetName']
            datasetName = checkDatasetName(datasetName, 0)
            evaluator = ADLStream.evaluation.PrequentialEvaluator(
                chunk_size=chunkSize,
                metric=metric,
                fadding_factor=faddingFactor,
                results_file="static/results/" + datasetName + " Results.csv",
                dataset_name=datasetName,
                show_plot=True,
                plot_file="static/test.jpg"
            )
            evaluatorParameters = {"interleavedChunkEvaluator": {
                "chunkSize": chunkSize,
                "metric": metric,
                "faddingFactor": faddingFactor
            }}
            parameterValues.update(evaluatorParameters)
            variables.update({"evaluator": evaluator,
                              "datasetName": datasetName})
            return redirect("/selectModel")
    else:
        return render_template('evaluators/selectEvaluator.html')


def checkDatasetName(dataset, i):
    if i != 0:
        dataset = dataset + str(i)
    if Results.query.filter(Results.fileName == dataset).first() != None:
        i = i + 1
        return checkDatasetName(dataset, i)
    else:
        res = dataset
        return res


# @app.route('/evaluatorForm', methods=['GET', 'POST'])
# def evaluatorForm():
#     evaluatorType = variables["evaluatorType"]
#     if evaluatorType == "interleavedChunkEvaluator":
#         if request.method == 'GET':
#             if variables["streamName"] == "CSVFileStream":
#                 dataSet = variables["datasetName"]
#             else:
#                 dataSet = "Fake Stream"
#             return render_template('evaluators/interleavedChunkEvaluatorForm.html', dataSet=dataSet)
#         else:
#             chunkSize = int(request.form['chunkSize'])
#             metric = request.form['metric']
#             datasetName = request.form['datasetName']
#             evaluator = ADLStream.evaluation.InterleavedChunkEvaluator(
#                 chunk_size=chunkSize,
#                 metric=metric,
#                 results_file="static/results/" + datasetName + " Results.csv",
#                 dataset_name=datasetName,
#                 show_plot=True,
#                 plot_file="static/test.jpg"
#
#             )
#             variables.update({"evaluator": evaluator, "datasetName": datasetName})
#             return redirect("/selectModel")
#     else:
#         if request.method == 'GET':
#             if variables["streamName"] == "CSVFileStream":
#                 dataSet = variables["datasetName"]
#             else:
#                 dataSet = "Fake Stream"
#             return render_template('evaluators/prequentialEvaluatorForm.html', dataSet=dataSet)
#         else:
#             chunkSize = int(request.form['chunkSize'])
#             faddingFactor = float(request.form['faddingFactor'])
#             metric = request.form['metric']
#             datasetName = request.form['datasetName']
#             evaluator = ADLStream.evaluation.PrequentialEvaluator(
#                 chunk_size=chunkSize,
#                 metric=metric,
#                 fadding_factor=faddingFactor,
#                 results_file="static/results/" + datasetName + " Results.csv",
#                 dataset_name=datasetName,
#                 show_plot=True,
#                 plot_file="static/test.jpg"
#             )
#             variables.update({"evaluator": evaluator,
#                               "datasetName": datasetName})
#             return redirect("/selectModel")


@app.route('/selectModel', methods=['GET', 'POST'])
def selectModel():
    if request.method == 'POST':
        model = request.form['model']
        return calculaParametros(model)
    else:
        if variables["streamName"] == "CSVFileStream":
            return render_template('models/selectModel.html')
        else:
            return render_template('models/selectModelFake.html')


# @app.route('/modelForm', methods=['GET', 'POST'])
# def modelForm():
#     modelName = variables["modelName"]
#     if request.method == 'GET':
#         return render_template('models/' + modelName + 'Form.html')
#     else:
#         return calculaParametros(modelName)
# if modelName == "mlp":
#    if request.method == 'GET':
#        return render_template('models/mlpForm.html')
#    else:
#        variables.update({"model": "mlp"})
#        parameters = {}
#        variables.update(({"parameters": parameters}))
#        return redirect("/selectLossOpt")
# elif modelName == "lstm":
#    if request.method == 'GET':
#        return render_template('models/lstmForm.html')
#    else:
#         variables.update({"model": "lstm"})
#         recurrentUnits = [int(x) for x in request.form["recurrentUnits"].split(",") if x != ""]
#         denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
#         recurrentDropout = int(request.form["dropout"])
#         returnSequences = request.form["returnSequences"]
#         denseDropout = float(request.form["denseDropout"])
#         denseActivation = request.form["denseActivation"]
#         outActivation = request.form["outActivation"]
#         parameters = {"recurrent_units": recurrentUnits,
#                       "recurrent_dropout": recurrentDropout,
#                       "return_sequences": returnSequences,
#                       "dense_layers": denseLayers,
#                       "dense_dropout": denseDropout,
#                       "dense_activation": denseActivation,
#                       "out_activation": outActivation
#                       }
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# elif modelName == "gru":
#     if request.method == 'GET':
#         return render_template('models/gruForm.html')
#     else:
#         variables.update({"model": "gru"})
#         parameters = {}
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# elif modelName == "ernn":
#     if request.method == 'GET':
#         return render_template('models/ernnForm.html')
#     else:
#         variables.update({"model": "ernn"})
#         parameters = {}
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# elif modelName == "esn":
#     if request.method == 'GET':
#         return render_template('models/esnForm.html')
#     else:
#         variables.update({"model": "esn"})
#         parameters = {}
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# elif modelName == "cnn":
#     if request.method == 'GET':
#         return render_template('models/cnnForm.html')
#     else:
#         variables.update({"model": "cnn"})
#         convLayers = [int(x) for x in request.form["convLayers"].split(",") if x != ""]
#         kernelSizes = [int(x) for x in request.form["kernelSizes"].split(",") if x != ""]
#         poolSizes = [int(x) for x in request.form["poolSizes"].split(",") if x != ""]
#         denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
#         denseDropout = float(request.form['denseDropout'])
#         activation = request.form['activation']
#         denseActivation = request.form['denseActivation']
#         outActivation = request.form['outActivation']
#         parameters = {
#             "conv_layers": convLayers,
#             "kernel_sizes": kernelSizes,
#             "pool_sizes": poolSizes,
#             "dense_layers": denseLayers,
#             "dense_dropout": denseDropout,
#             "activation": activation,
#             "dense_activation": denseActivation,
#             "out_activation": outActivation
#         }
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# elif modelName == "tcn":
#     if request.method == 'GET':
#         return render_template('models/tcnForm.html')
#     else:
#         variables.update({"model": "tcn"})
#         denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
#         parameters = {}
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")
# else:
#     if request.method == 'GET':
#         return render_template('models/transformerForm.html')
#     else:
#         variables.update({"model": "transformer"})
#         parameters = {}
#         variables.update(({"parameters": parameters}))
#         return redirect("/selectLossOpt")


@app.route('/lastParameters', methods=['GET', 'POST'])
def selectLossOpt():
    if request.method == 'POST':
        loss = request.form['loss']
        batchSize = int(request.form['batchSize'])
        numBatchesFed = int(request.form['numBatchesFed'])
        opt = request.form['opt']
        variables.update({"loss": loss})
        variables.update({"opt": opt})
        variables.update({"batchSize": batchSize})
        variables.update({"numBatchesFed": numBatchesFed})
        modelParameters = {"Last Parameters": {
            "loss":loss,
            "opt": opt,
            "batchSize": batchSize,
            "numBatchesFed": numBatchesFed,
        }}
        parameterValues.update(modelParameters)
        return redirect("/execute")
    return render_template('lastParameters.html')


@app.route('/execute', methods=['GET', 'POST'])
def execADLStream():
    if request.method == 'GET':
        streamName = variables["streamName"]
        stream = variables["stream"]
        stream_generator = variables["generator"]
        evaluator = variables["evaluator"]
        model = variables["model"]
        parameters = variables["parameters"]
        loss = variables["loss"]
        optimizer = variables["opt"]
        batchSize = variables["batchSize"]
        numBatchesFed = variables["numBatchesFed"]
        adls = ADLStream.ADLStream(
            stream_generator=stream_generator,
            evaluator=evaluator,
            batch_size=batchSize,
            num_batches_fed=numBatchesFed,
            model_architecture=model,
            model_loss=loss,
            model_optimizer=optimizer,
            model_parameters=parameters,
            log_file="ADLStream.log"
        )
        adls.run()
        datasetName = variables["datasetName"]
        evaluatorType = variables["evaluatorType"]
        parameterJson = json.dumps(parameterValues)
        f = open("parameters/" + datasetName + " Parameters.txt", "w")
        f.write(parameterJson)
        f.close()
        newResult = Results(datasetName, model, streamName, evaluatorType)
        db_session.add(newResult)
        db_session.commit()
        if streamName != "fakeStream":
            os.remove(stream.filename)
        return render_template('result.html', testImage="static/test.jpg")
    else:
        datasetName = variables["datasetName"]
        return send_file("static/results/" + datasetName + " Results.csv", as_attachment=True)


@app.route('/history', methods=['GET', 'POST'])
def historial():
    if request.method == 'GET':
        BASE_DIR = 'static/results'
        abs_path = os.path.join(BASE_DIR)

        if os.path.isfile(abs_path):
            return send_file(abs_path)

        files = Results.query.all()
        return render_template('history.html', files=files)
    else:
        fileName = request.form['fileName']
        file = "parameters/" + fileName + " Parameters.txt"
        if request.form['download'] == "Results":
            file = "static/results/" + fileName + " Results.csv"
        return send_file(file, as_attachment=True)


def calculaParametros(modelName):
    variables.update({"model": modelName})
    if modelName == "mlp":
        hiddenLayers = [int(x) for x in request.form["hiddenLayers"].split(",") if x != ""]
        dropout = float(request.form["dropout"])
        activation = request.form["activation"]
        outActivation = request.form["outActivation"]
        parameters = {"hidden_layers": hiddenLayers,
                      "dropout": dropout,
                      "activation": activation,
                      "out_activation": outActivation
                      }
    elif modelName == "lstm":
        recurrentUnits = [int(x) for x in request.form["recurrentUnits"].split(",") if x != ""]
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        recurrentDropout = int(request.form["dropout"])
        returnSequences = request.form["returnSequences"]
        denseDropout = float(request.form["denseDropout"])
        denseActivation = request.form["denseActivation"]
        outActivation = request.form["outActivation"]
        parameters = {"recurrent_units": recurrentUnits,
                      "recurrent_dropout": recurrentDropout,
                      "return_sequences": returnSequences,
                      "dense_layers": denseLayers,
                      "dense_dropout": denseDropout,
                      "dense_activation": denseActivation,
                      "out_activation": outActivation
                      }
    elif modelName == "gru":
        recurrentUnits = [int(x) for x in request.form["recurrentUnits"].split(",") if x != ""]
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        recurrentDropout = int(request.form["dropout"])
        returnSequences = request.form["returnSequences"]
        denseDropout = float(request.form["denseDropout"])
        outActivation = request.form["outActivation"]
        parameters = {"recurrent_units": recurrentUnits,
                      "recurrent_dropout": recurrentDropout,
                      "return_sequences": returnSequences,
                      "dense_layers": denseLayers,
                      "dense_dropout": denseDropout,
                      "out_activation": outActivation
                      }
    elif modelName == "ernn":
        recurrentUnits = [int(x) for x in request.form["recurrentUnits"].split(",") if x != ""]
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        recurrentDropout = int(request.form["dropout"])
        returnSequences = request.form["returnSequences"]
        denseDropout = float(request.form["denseDropout"])
        outActivation = request.form["outActivation"]
        parameters = {"recurrent_units": recurrentUnits,
                      "recurrent_dropout": recurrentDropout,
                      "return_sequences": returnSequences,
                      "dense_layers": denseLayers,
                      "dense_dropout": denseDropout,
                      "out_activation": outActivation
                      }
    elif modelName == "esn":
        recurrentUnits = [int(x) for x in request.form["recurrentUnits"].split(",") if x != ""]
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        recurrentDropout = int(request.form["dropout"])
        returnSequences = request.form["returnSequences"]
        denseDropout = float(request.form["denseDropout"])
        outActivation = request.form["outActivation"]
        parameters = {"recurrent_units": recurrentUnits,
                      "recurrent_dropout": recurrentDropout,
                      "return_sequences": returnSequences,
                      "dense_layers": denseLayers,
                      "dense_dropout": denseDropout,
                      "out_activation": outActivation
                      }
    elif modelName == "cnn":
        convLayers = [int(x) for x in request.form["convLayers"].split(",") if x != ""]
        kernelSizes = [int(x) for x in request.form["kernelSizes"].split(",") if x != ""]
        poolSizes = [int(x) for x in request.form["poolSizes"].split(",") if x != ""]
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        denseDropout = float(request.form["denseDropout"])
        activation = request.form["activation"]
        denseActivation = request.form["denseActivation"]
        outActivation = request.form["outActivation"]
        parameters = {
            "conv_layers": convLayers,
            "kernel_sizes": kernelSizes,
            "pool_sizes": poolSizes,
            "dense_layers": denseLayers,
            "dense_dropout": denseDropout,
            "activation": activation,
            "dense_activation": denseActivation,
            "out_activation": outActivation
        }
    elif modelName == "tcn":
        nbFilters = int(request.form["nbFilters"])
        kernelSize = int(request.form["kernelSize"])
        nbStacks = int(request.form["nbStacks"])
        dilations = [int(x) for x in request.form["dilations"].split(",") if x != ""]
        tcnDropout = float(request.form["tcnDropout"])
        returnSequences = bool(request.form["returnSequences"])
        activation = request.form["activation"]
        denseActivation = request.form["denseActivation"]
        outActivation = request.form["outActivation"]
        padding = request.form["padding"]
        skipConnections = bool(request.form["skipConnections"])
        barchNorm = bool(request.form["barchNorm"])
        denseLayers = [int(x) for x in request.form["denseLayers"].split(",") if x != ""]
        denseDropout = int(request.form["denseDropout"])
        parameters = {"nb_filters": nbFilters,
                      "kernel_size": kernelSize,
                      "nb_stacks": nbStacks,
                      "dense_layers": denseLayers,
                      "dense_dropout": denseDropout,
                      "activation": activation,
                      "dense_activation": denseActivation,
                      "out_activation": outActivation,
                      "dilations": dilations,
                      "tcn_dropout": tcnDropout,
                      "return_sequences": returnSequences,
                      "padding": padding,
                      "use_skip_connections": skipConnections,
                      "use_batch_norm": barchNorm
                      }
    else:
        forecastingHorizon = variables.get("forecastingHorizon")
        attribute = [x for x in request.form["attribute"].split(",") if x != ""]
        numHeads = int(request.form["numHeads"])
        numLayers = int(request.form["numLayers"])
        dModel = int(request.form["dModel"])
        dff = int(request.form["dff"])
        peInput = int(request.form["peInput"])
        dropoutRate = float(request.form["dropoutRate"])
        activation = request.form["activation"]
        parameters = {"output_shape": [forecastingHorizon, 1],
                      "attribute": attribute,
                      "num_heads": numHeads,
                      "num_layers": numLayers,
                      "d_model": dModel,
                      "dff": dff,
                      "pe_input": peInput,
                      # "pe_output":peOutput,
                      "dropout_rate": dropoutRate,
                      "activation": activation}
    modelParameters = {modelName: parameters}
    parameterValues.update(modelParameters)
    variables.update(({"parameters": parameters}))
    return redirect("/lastParameters")


if __name__ == '__main__':
    app.run()
