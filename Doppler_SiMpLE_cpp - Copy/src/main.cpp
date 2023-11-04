#include "objectiveFunction.h"
#include "dopplerObjectiveFunction.h"

int main(int argc, char* argv[])
{
    // ------------------------------------------------------------------------
    // ARGUMENT PARSING
    // ------------------------------------------------------------------------
    params config;
    int err = parseArgs(&config, argc, argv);
    if (err) exit(1);

    // ------------------------------------------------------------------------
    // LOAD THE SCANS PATHS
    // ------------------------------------------------------------------------
    std::vector<std::string> scanFiles;
    for (auto const& dir_entry : std::filesystem::directory_iterator(config.path)) 
        scanFiles.push_back(dir_entry.path());

    // sort the scans in order of the file name
    std::sort(scanFiles.begin(), scanFiles.end(), compareStrings);
    // number of scans
    unsigned int numScans = scanFiles.size();

    //-------------------------------------------------------------------------
    // LOAD THE GROUND TRUTH PATH
    //-------------------------------------------------------------------------
    std::string gtPath = config.gtpath;
    Eigen::MatrixXd gtPoses = readGtFile(gtPath);

    //-------------------------------------------------------------------------
    // LOAD THE CALIBRATION PATH
    //-------------------------------------------------------------------------
    std::string calibPath = config.calibpath;
    Eigen::Matrix4d Tvehicle_aeva = readCalibFile(calibPath);

    // ------------------------------------------------------------------------
    // DETERMINE POINT CLOUD REGISTRATION RESULTS
    // ------------------------------------------------------------------------     

    // store the pose estimates in (roll,pitch,yaw,x,y,z,registrationScore) format
    std::vector<std::vector<double> > poseEstimates(numScans,std::vector<double>(7));

    // initialise the seeds as zero
    std::vector<double> rotationalSeed = {0.0, 0.0, 0.0};
    std::vector<double> translationalSeed = {0.0, 0.0, 0.0};
    std::vector<double> seed = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


    // start the timer
    auto startReg = std::chrono::high_resolution_clock::now();

    // loop over all input scans, update the submap, and save the registration result
    for (unsigned int scanNum = 0; scanNum < numScans; scanNum++) //numScans
    {
        // --------------------------------------------------------------------
        // READ THE POINT CLOUD FROM THE .bin FILES (AEVA FORMAT)
        // --------------------------------------------------------------------
        // read the input scan
        std::set<int> allPoints;
        std::vector<double> scanVel;
        Eigen::MatrixXd ptsInRange = readScan(scanFiles[scanNum], config, allPoints, scanVel);           
        // --------------------------------------------------------------------
        // STEP 1: SUBSAMPLE THE INPUT POINT CLOUD AT rNew
        // --------------------------------------------------------------------
        std::vector<double> scanVelSs;  
        Eigen::MatrixXd ptsSubsampled = subsample(config.rNew, allPoints, ptsInRange, scanVel, scanVelSs);

        
        
/////////////////////////////////////////////////////////////////////////////////////////////////
        // --------------------------------------------------------------------
        // STEP 2: INPUT POINT CLOUD TO LOCAL MAP REGISTRATION
        // --------------------------------------------------------------------
        if (scanNum > 0)
        {
            // --------------------------------------------------------------------
            // GET THE GROUND TRUTH SCAN TO SCAN POSE FOR THE SCAN
            // --------------------------------------------------------------------
            Eigen::Matrix4d TW_2;
            TW_2.setZero();
            TW_2(0,0) = gtPoses(scanNum, 0);
            TW_2(0,1) = gtPoses(scanNum, 1);
            TW_2(0,2) = gtPoses(scanNum, 2);
            TW_2(0,3) = gtPoses(scanNum, 3);
            TW_2(1,0) = gtPoses(scanNum, 4);
            TW_2(1,1) = gtPoses(scanNum, 5);
            TW_2(1,2) = gtPoses(scanNum, 6);
            TW_2(1,3) = gtPoses(scanNum, 7);
            TW_2(2,0) = gtPoses(scanNum, 8);
            TW_2(2,1) = gtPoses(scanNum, 9);
            TW_2(2,2) = gtPoses(scanNum, 10);
            TW_2(2,3) = gtPoses(scanNum, 11);
            TW_2(3,3) = 1;

            Eigen::Matrix4d TW_1;
            TW_1.setZero();
            TW_1(0,0) = gtPoses(scanNum-1, 0);
            TW_1(0,1) = gtPoses(scanNum-1, 1);
            TW_1(0,2) = gtPoses(scanNum-1, 2);
            TW_1(0,3) = gtPoses(scanNum-1, 3);
            TW_1(1,0) = gtPoses(scanNum-1, 4);
            TW_1(1,1) = gtPoses(scanNum-1, 5);
            TW_1(1,2) = gtPoses(scanNum-1, 6);
            TW_1(1,3) = gtPoses(scanNum-1, 7);
            TW_1(2,0) = gtPoses(scanNum-1, 8);
            TW_1(2,1) = gtPoses(scanNum-1, 9);
            TW_1(2,2) = gtPoses(scanNum-1, 10);
            TW_1(2,3) = gtPoses(scanNum-1, 11);
            TW_1(3,3) = 1;

            Eigen::Matrix4d T1_2 = TW_1.inverse()*TW_2;
            std::vector<double> seedGtDoppler = hom2rpyxyz(T1_2);

            // -----------------------------------------------------------------------------------------------------------
            // DOPPLER REGISTRATION
            // -----------------------------------------------------------------------------------------------------------
            // instantiate the objective function
            DopplerObjectiveFunction doppObjFunc = DopplerObjectiveFunction(config.sigma, ptsSubsampled, scanVelSs, Tvehicle_aeva);

            // set initial seed for scan
            column_vector doppRegResult = {rotationalSeed[0], rotationalSeed[1], rotationalSeed[2],
                                translationalSeed[0], translationalSeed[1], translationalSeed[2]};


            // if you want to seed to be ground truth
            // column_vector doppRegResult = {seedGtDoppler[0], seedGtDoppler[1], seedGtDoppler[2],
            //                     seedGtDoppler[3], seedGtDoppler[4], seedGtDoppler[5]};          
            // find the best solution to the objective function
            double doppRegistrationScore = dlib::find_min_using_approximate_derivatives(
                                    dlib::bfgs_search_strategy(),
                                    dlib::objective_delta_stop_strategy(config.convergenceTol),
                                    doppObjFunc, doppRegResult, -10000000); //FMINUNC
            // save the results
            poseEstimates[scanNum] = {doppRegResult(0), doppRegResult(1), doppRegResult(2),
                                          doppRegResult(3), doppRegResult(4), doppRegResult(5), doppRegistrationScore};

            // rotational seed set to be zero for next scan
            rotationalSeed = {0, 0, 0};
            

            // translational seed set to be previous estimate for next scan
            translationalSeed = {poseEstimates[scanNum][3], poseEstimates[scanNum][4], poseEstimates[scanNum][5]};


        }
        // print the progress to the terminal
        // comment printProgress to remove this 
        printProgress((double(scanNum) / numScans));
        
    }
    printf("\n"); // end with a new line character for the progress bar

    // calculate the average time per registration result
    auto stopReg = std::chrono::high_resolution_clock::now();
    auto durationReg = std::chrono::duration_cast<std::chrono::milliseconds>(stopReg - startReg);
    double avgTimePerScan = durationReg.count() / numScans;

    // ------------------------------------------------------------------------
    // OUTPUT RESULTS
    // ------------------------------------------------------------------------
    if (config.verbose)
        std::cout << "avgTimePerScan [ms] = " << avgTimePerScan << ";" << std::endl;

    // output a file with the results in the KITTI format, and a file with the configuration parameters
    writeResults(&config, poseEstimates, config.outputFileName, avgTimePerScan);

    return 0;
}