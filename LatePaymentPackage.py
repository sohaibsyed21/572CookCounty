"""
Name: Classification for Late Payment of Property Tax Payers
Creation Date: Jul 26, 2023
Authors: Shashank Parameswaran, Tinh Cao, Chris Chen, Sohail Syed, Zainab Hasnain
Organization: Illinois Institute of Technology
(C) All Rights Reserved
"""


# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp, log
from pyspark.sql import Window
import pyspark.sql.functions as f
from pyspark.sql import Window
from pyspark.sql.types import StringType, DateType, FloatType
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
#from pyspark_dist_explore import hist
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
import os
import findspark
findspark.init()
findspark.find()

# Create a spark session
spark = SparkSession.builder\
                    .master("local")\
                    .appName("Colab")\
                    .config('spark.ui.port', '4051')\
                    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
display(HTML("<style>pre { white-space: pre !important; }</style>"))


class LatePaymentClassifier:
    
    # Initialize variables
    def __init__(self,
                 file1="2017.rpt", # Year1 Master data 
                 file2="2018.rpt", # Year2 Master data
                 file3="2019.rpt", # Year3 Master data
                 file4="2020.rpt", # Year4 Master data
                 fileCurrentYear="2021.rpt", # Current Year data - to be predicted
                 fileHeader="Header.rpt",  # Header file
                 fileDueDates="InstDates.csv", # Updated Tax Due dates csv file
                 pmtFile1="TY2017.rpt", # Year1 Payment Master data
                 pmtFile2="TY2018.rpt", # Year2 Payment Master data
                 pmtFile3="TY2019.rpt", # Year3 Payment Master data
                 pmtFile4="TY2020.rpt", # Year4 Payment Master data
                 pmtCurrentYear="TY2021.rpt", # Current Year data
                 currentYear = 2021 # Enter current year
                ):
        
        self.data = None
        self.pmt = None
        self.train = None
        self.test = None
        self.predictionsDf = None
        self.actual = None
        self.predictions = None
        self.model = None
        self.results = None
        self.featureImportanceDf = None
        self.gainLiftDf = None
        self.element=f.udf(lambda v:float(v[1]),FloatType())
        
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.fileCurrentYear = fileCurrentYear
        self.fileHeader = fileHeader
        self.fileDueDates = fileDueDates
        self.pmtFile1 = pmtFile1
        self.pmtFile2 = pmtFile2
        self.pmtFile3 = pmtFile3
        self.pmtFile4 = pmtFile4
        self.pmtCurrentYear = pmtCurrentYear
        self.currentYear = currentYear
    
    # Read data files and create target variable
    def readData(self):
        print("Reading Data.....")
        data = spark.read.csv(self.fileCurrentYear, header=None, sep="|")
        header = pd.read_csv(self.fileHeader, sep="|")
        cols = header.columns
        data = data.toDF(*cols)
        load_data1 = spark.read.csv(self.file1, header=None, sep="|")
        load_data2 = spark.read.csv(self.file2, header=None, sep="|")
        load_data3 = spark.read.csv(self.file3, header=None, sep="|")
        load_data4 = spark.read.csv(self.file4, header=None, sep="|")
        
        data = data.union(load_data1)
        data = data.union(load_data2)
        data = data.union(load_data3)
        data = data.union(load_data4)
        
        print("Initial property master dataset size:", [data.count(), len(data.columns)])
        
        numeric_cols = ["AdjustedAmountDue1", "TaxAmountDue1", "InterestAmountDue1", "CostAmountDue1", "TotalAmountDue1", "OriginalTaxDue2",
        "AdjustedTaxDue2", "TaxAmountDue2", "InterestAmountDue2", "CostAmountDue2", "TotalAmountDue2", "AssessedValuation",
        "EqualizedEvaluation", "TaxRate", "LastPaymentReceivedAmount1", "LastPaymentReceivedAmount2"
        ] + list(data.columns)[70:81]
        
        # Convert to numeric columns
        for col_name in numeric_cols:
            data = data.withColumn(col_name, col(col_name).cast('float'))

        # Remove 3 rows which have garbage values
        data = data.where(data.SegmentCode=="PH")
        
        instDates = spark.read.csv(self.fileDueDates, header=True)
        instDates = instDates.withColumn("Year", col("Year").cast("int"))
        instDates = instDates.where(instDates.Year>=2017)
        instDates = instDates.withColumnRenamed("Inst1", "PmtDue1")
        instDates = instDates.withColumnRenamed("Inst2", "PmtDue2")
        
        # Conditions
        data = data.where(data.TaxStatus!="01")
        TPA_pmt_list = ['527', '600', '800', '802', '830']
        data = data.where(~((data.AdjustedAmountDue1==0) & (data.AdjustedTaxDue2==0)))
        data = data.where(~data.LastPaymentSource2.isin(TPA_pmt_list))
        data = data.where(~data.LastPaymentSource1.isin(TPA_pmt_list))
        
        data = data.join(instDates, data.TaxYear==instDates.Year, "left")
        data = data.withColumn("LastPaymentDate1", f.when(data.LastPaymentDate1=="00000000", "12312099").otherwise(data.LastPaymentDate1))
        data = data.withColumn("LastPaymentDate2", f.when(data.LastPaymentDate2=="00000000", "12312099").otherwise(data.LastPaymentDate2))
        data = data.withColumn("LastPaymentDate1", f.to_date(col("LastPaymentDate1"), "MMddyyyy"))
        data = data.withColumn("LastPaymentDate2", f.to_date(col("LastPaymentDate2"), "MMddyyyy"))
        data = data.withColumn("PmtDue1", f.to_date(col("PmtDue1"), "yyyy-MM-dd"))
        data = data.withColumn("PmtDue2", f.to_date(col("PmtDue2"), "yyyy-MM-dd"))
        data = data.withColumn("LatePmt1", f.when(col("LastPaymentDate1")>col("PmtDue1"), 1).otherwise(0))
        data = data.withColumn("LatePmt2", f.when(col("LastPaymentDate2")>col("PmtDue2"), 1).otherwise(0))
        data = data.withColumn("BlankPmt1", f.when(col("LastPaymentDate1")=="2099-12-31", 1).otherwise(0))
        data = data.withColumn("BlankPmt2", f.when(col("LastPaymentDate2")=="2099-12-31", 1).otherwise(0))
        
        data = data.withColumn("LastPaymentDate1", f.when(data.LastPaymentDate1=="2099-12-31", f.current_date()) \
                       .otherwise(data.LastPaymentDate1))
        data = data.withColumn("LastPaymentDate2", f.when(data.LastPaymentDate2=="2099-12-31", f.current_date()) \
                               .otherwise(data.LastPaymentDate2))
        data = data.withColumn("DiffPmt1", f.datediff(col("LastPaymentDate1"), col("PmtDue1")))
        data = data.withColumn("DiffPmt1", f.when(col("DiffPmt1")<0,0).otherwise(col("DiffPmt1")))
        data = data.withColumn("DiffPmt2", f.datediff(col("LastPaymentDate2"), col("PmtDue2")))
        
        data = data.where(~((data.DiffPmt2>=500)))
        data = data.where(~((data.DiffPmt1>=680)))
        
        pmt = spark.read.csv(self.pmtCurrentYear, sep=",", header=True)
        pmt_data1 = spark.read.csv(self.pmtFile1, sep=",", header=True)
        pmt_data2 = spark.read.csv(self.pmtFile2, sep=",", header=True)
        pmt_data3 = spark.read.csv(self.pmtFile3, sep=",", header=True)
        pmt_data4 = spark.read.csv(self.pmtFile4, sep=",", header=True)
        
        pmt = pmt.union(pmt_data1)
        pmt = pmt.union(pmt_data2)
        pmt = pmt.union(pmt_data3)
        pmt = pmt.union(pmt_data4)
        
        drop_cols_inst2 = data.columns[42:53] + ["LastPaymentSource2"]
        drop_cols_unrelated = ["SegmentCode", "TaxpayerName", "TaxpayerMailingAddress", "TaxpayerMailingZip",
                               "TaxpayerPropertyHouse","TaxpayerPropertyDirection", "TaxpayerPropertyStreet",
                               "TaxpayerPropertyZip", "RecordCount"
                               ]
        drop_cols_univariate = ["LongtimeHomeownersExempt", "TaxInfoType", "TaxType", "TaxpayerPropertyState", "BillYear", "SegmentCode2",
                                "InstallmentNumber1", "RefundTaxAmountDueIndicator1", "RefundInterestDueIndicator1",
                                "RefundTotalDueIndicator1", "RefundCostDueIndicator1", "EndMarker", "TaxpayerMailingCity",
                                "BankruptStatus", "AcquisitionStatus", "CondemnationStatus", "RefundStatus"
                                ]
        drop_cols_numeric_corr = ["OriginalTaxDue2", "AdjustedTaxDue2", "AssessedValuation", "EqualizedEvaluation",
                                  "LastPaymentReceivedAmount2", "TaxDueEstimated1", "LastPaymentReceivedAmount1",
                                  "AdjustedAmountDue1_Org", "InterestAmountDue1", "TotalAmountDue1", "InterestAmountDue2",
                                  "TotalAmountDue2", "SeniorExemptAmount", "VeteranExempt"]
        drop_cols_rf_imp = ["DisabledPersonVetExemptionAmount", "DisabledVetExemptionAmount", "DisabledPersonExemptionAmount",
                            "ReturningVetExemptionAmount", "SeniorFreezeExempt", "MunicipalAcquisitionStatus",
                            "TaxpayerMailingState", "CofENumber", "Volume"
                            ]
        
        data = data.drop(*drop_cols_inst2)
        data = data.drop(*drop_cols_unrelated)
        data = data.drop(*drop_cols_univariate)
        data = data.drop(*drop_cols_numeric_corr)
        data = data.drop(*drop_cols_rf_imp)
        
        drop_cols_pmt_unrelated = ["TaxPayer", "SerialNumber", "RefundNumber", "Volume", "TaxYear",
                            "WarrantYear", "TaxType", "DateUpdated", "TaxPaid", "InterestPaid", "CostPaid"
                            ]
        pmt = pmt.drop(*drop_cols_pmt_unrelated)
        
        self.data = data
        self.pmt = pmt
    
    # Run some simple transformations
    def simpleTransform(self):
        print("Running simple transformations.....")
        data1 = self.data
        data1 = data1.withColumn("TaxpayerPropertyCity", f.trim(data1.TaxpayerPropertyCity))
        data1 = data1.withColumn("TaxpayerPropertyCity", f.when(data1.TaxpayerPropertyCity=="CHICAGO", 1).otherwise(0))
        data1 = data1.withColumn("BillType", f.when(data1.BillType=="1", 1).otherwise(0))

        data1 = data1.withColumn("TaxStatus", col("TaxStatus").cast("int"))
        data1 = data1.withColumn("PastTaxSaleStatus", f.when(col("PastTaxSaleStatus")=="Y", 1).otherwise(0))
        
        pmt = self.pmt
        pmt = pmt.where(col("Payment")=="P1")
        pmt = pmt.withColumn("DatePaid", f.trim(col("DatePaid")))
        pmt = pmt.withColumn("DatePaid", f.to_date(col("DatePaid"), "yyMMdd"))
        pmt = pmt.groupBy(["PIN", "TaxYear4"]).agg(
                                            f.count("Payment").alias("PaymentCnt"),
                                            f.max("DatePaid").alias("LastPaid1"),
                                            f.countDistinct("SourceID").alias("SourceIDCnt"),
                                            f.sum("TotalPaid").alias("TotalPaid")
                                            )
        pmt = pmt.withColumn("PaymentCnt", f.when(col("PaymentCnt")==1,1).otherwise(0))
        pmt = pmt.withColumn("SourceIDCnt", f.when(col("SourceIDCnt")==1,1).otherwise(0))
        pmt = pmt.withColumnRenamed("TaxYear4", "TaxYear")
        pmt = pmt.withColumn("TotalPaid", f.round(col("TotalPaid"),2))
        pmt = pmt.persist()
        #print("Payment master size:", [pmt.count(), len(pmt.columns)])
        
        data1 = data1.join(pmt, ["PIN", "TaxYear"], "left")
        self.pmt = pmt
        self.data = data1.persist()
        
        print("Dataset size after simple transformation:", [self.data.count(), len(self.data.columns)])
    
    # WOE function to convert categorical variable to its WOE form
    def woe(self, df, var, y="LatePmt2"):
        woe_df = df.groupBy(var) \
            .agg(f.count(var).alias("cnt"), \
                f.sum(y).alias("cnt_1")) \
            .withColumn("cnt_0", col("cnt") - col("cnt_1")) \
            .withColumn("perc_1", col("cnt_1")/col("cnt")) \
            .withColumn("perc_0", col("cnt_0")/col("cnt")) \
            .withColumn("WOE_"+var, f.log(col("perc_0")/col("perc_1")))
        woe_df = woe_df.withColumn("WOE_"+var, f.round(col("WOE_"+var), 2))
        df = df.join(woe_df.select(*[var, "WOE_"+var]), [var])#.drop(woe_df[var])
        return(df)
    
    # Run the WOE transformations
    def woeTransform(self):
        print("Running WOE transformations.....")
        data2 = self.data
        woe_transform_cols = [
                            "TaxpayerPropertySuffix", "Classification", "TaxpayerPropertyTown",
                            "TaxStatus", "AccountStatus", "LastPaymentSource1"
                            ]
        for var in woe_transform_cols:
            data2 = self.woe(data2, var)
        data2 = data2.drop(*woe_transform_cols)
        self.data = data2.persist()
        print("Dataset size after WOE Transformation:", [self.data.count(), len(self.data.columns)])
    
    # Run the window transformations
    def windowTransform(self):
        print("Running window transformations.....")
        data3 = self.data
        windowval1 = (Window.partitionBy('PIN').orderBy('TaxYear')
                     .rangeBetween(Window.unboundedPreceding, 0))
        data3 = data3.withColumn('PastTaxSaleStatusSum', f.sum('PastTaxSaleStatus').over(windowval1))
        windowval2 = (Window.partitionBy('PIN').orderBy('TaxYear')
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow-1))
        data3 = data3.withColumn('LatePmt2Sum', f.sum('LatePmt2').over(windowval2))
        data3 = data3.na.fill(value=0,subset=["LatePmt2Sum"])

        windowval3 = (Window.partitionBy('PIN').orderBy('TaxYear')
                    .rowsBetween(Window.unboundedPreceding, 0))
        data3 = data3.withColumn('LatePmt1Sum', f.sum('LatePmt1').over(windowval3))
        self.data = data3.persist()
        print("Dataset size after window transformation:", [self.data.count(), len(self.data.columns)])
    
    # Few other transformations
    def otherTransform(self):
        print("Running few other transformations.....")
        data4 = self.data
        other_cols = ["LastPaymentDate1", "LastPaymentDate2", "PmtDue1", "PmtDue2", 
                      "Year", "DiffPmt2", "BlankPmt1", "BlankPmt2",
                      "LastPaid1", "TotalPaid", "TaxYear"
                     ]
        data4 = data4.drop(*other_cols)
        string_cols = []
        for c in data4.schema.fields:
            if isinstance(c.dataType, StringType):
                string_cols.append(c.name)

        string_cols.remove("PIN")
        for c in string_cols:
            data4 = data4.withColumn(c, data4[c].cast('int'))    
        data4 = data4.withColumn("PIN", data4.PIN.cast("bigint"))
        
        # Null values are replaced with 0s. It can be improved (although not too many null values)
        null_df = data4.select([f.count(f.when(col(c).isNull(), c)).alias(c) for c in data4.columns])
        null_cols = [k for k,v in null_df.collect()[0].asDict().items() if v >0]
        
        for c in null_cols:
            data4 = data4.withColumn(c, f.when(data4[c].isNull(), 0).otherwise(data4[c]))
        self.data = data4.persist()
        print("Dataset size for model building:", [self.data.count(), len(self.data.columns)])
        
    # Accuracy, Precision, Recall and F1 Score are stored in 'results'
    def getResults(self):
        
        nums = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        acc = []
        f1 = []
        rec = []
        prec = []
        for num in nums:
            pred = np.where(self.predictions>num, 1, 0)
            acc.append(accuracy_score(self.actual, pred))
            f1.append(f1_score(self.actual, pred))
            rec.append(recall_score(self.actual, pred))
            prec.append(precision_score(self.actual, pred))
        res_df = pd.DataFrame(nums, columns=["Threshold"])
        res_df["Acc"] = acc
        res_df["F1"] = f1
        res_df["Rec"] = rec
        res_df["Prec"] = prec
        self.results = res_df
    
    # Split into train and test sets
    def trainTestSplit(self):
        print("Splitting into train and test data.....")
        data4 = self.data
        feature_cols = data4.columns
        feature_cols.remove("LatePmt2")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        data5 = assembler.transform(data4)
        test = data5.where(col("TaxYear")==self.currentYear)
        train = data5.filter(~(col("TaxYear")==self.currentYear))
        self.train = train.persist()
        self.test = test.persist()
        print("Dataset size for training:", [self.train.count(), len(self.train.columns)-1])
        print("Dataset size for testing:", [self.test.count(), len(self.test.columns)-1])
    
    # Fit the Random forest model
    def fit(self):
        print("Fitting the model.....")
        rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'LatePmt2', numTrees=50, maxDepth=8)
        rfModel = rf.fit(self.train)
        self.predictionsDf = rfModel.transform(self.test)
        self.actual = self.test.select(self.test.LatePmt2).toPandas()['LatePmt2']
        self.predictions = self.predictionsDf.select(["probability"]) \
                                             .withColumn("probability", f.round(self.element("probability"), 2)) \
                                             .toPandas()
        self.predictions = self.predictions["probability"].to_numpy()
        self.model = rfModel
    
    # Get the feature importance table
    def getFeatureImportance(self):
        print("Retreiving the most important features.....")
        feat_imp = self.model.featureImportances
        feat_imp = [c for c in feat_imp]
        train_cols = self.train.columns
        train_cols.remove("features")
        train_cols.remove("LatePmt2")
        feat_df = pd.DataFrame(train_cols, columns=["Features"])
        feat_df["Importance"] = feat_imp
        feat_df = feat_df.sort_values("Importance", ascending=False)
        self.featureImportanceDf = feat_df
    
    # create the Gain Lift dataframe
    def getGainLiftChart(self):
        print("Obtaining Gain-Lift charts.....")
        predictions_base2 = pd.DataFrame(self.predictions, columns=["probability"])
        predictions_base2['Actual'] = self.actual
        predictions_base2["DecileRank"] = pd.qcut(predictions_base2["probability"].rank(method='first', ascending=False), 
                                                  10, labels=np.arange(1,11))
        predictions_base3 = predictions_base2.groupby("DecileRank").agg(
                                                                        DecileN=("Actual", "count"),
                                                                        GainN=("Actual", "sum")
                                                                    ).reset_index()
        predictions_base3["CumulativeGainN"] = predictions_base3["GainN"].cumsum()
        predictions_base3["Late%ByDecile"] = predictions_base3["GainN"]/predictions_base3["DecileN"]*100
        predictions_base3["GainN%"] = predictions_base3["GainN"]/predictions_base3["GainN"].sum()*100
        predictions_base3["CumulativeGain%"] = predictions_base3["CumulativeGainN"]/predictions_base3["GainN"].sum()*100

        overall_late_rate = predictions_base2['Actual'].sum()/predictions_base2['Actual'].count()*100
        predictions_base3["Lift"] = predictions_base3["Late%ByDecile"]/overall_late_rate

        predictions_base3["DecileRank"] = predictions_base3["DecileRank"].astype("int")
        predictions_base3["CumulativeGain%"] = np.round(predictions_base3["CumulativeGain%"], 3)
        predictions_base3["GainN%"] = np.round(predictions_base3["GainN%"], 3)
        predictions_base3["Lift"] = np.round(predictions_base3["Lift"], 3)
        predictions_base3["Late%ByDecile"] = np.round(predictions_base3["Late%ByDecile"], 3)
        self.gainLiftDf = predictions_base3
    
    # Plot Gain Lift charts
    def plotGainLiftChart(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        ax[0].plot(self.gainLiftDf["DecileRank"], self.gainLiftDf["CumulativeGain%"])
        ax[0].plot(self.gainLiftDf["DecileRank"], self.gainLiftDf["DecileRank"]*10)
        ax[0].set_xlabel("Decile")
        ax[0].set_ylabel("CumulativeGain%")
        ax[0].set_title("Gain Chart")

        ax[1].plot(self.gainLiftDf["DecileRank"], self.gainLiftDf["Lift"])
        ax[1].set_title("Lift Chart")
        ax[1].set_xlabel("Decile")
        ax[1].set_ylabel("Lift")
        plt.show()
    
    # Export the predictions to a CSV file
    def exportToCSV(self):
        outDf = self.predictionsDf.select(*("PIN", "probability", "prediction"))
        outDf = outDf.withColumn("probability", f.round(self.element("probability"), 2))
        outDf.toPandas().to_csv("Predictions.csv")
        print("CSV Exported to:", os.getcwd()+"/"+"Predictions.csv")