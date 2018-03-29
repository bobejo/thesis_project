% Run the matlab camera calibrator from the computer vision toolbox
% Information how to use the tool can be found here https://mathworks.com/help/vision/ug/stereo-camera-calibrator-app.html

stereoCameraCalibrator
%% Create camera matrix
%  Create the camera matrix with the intrinsic and extrinsic matrices.
%  Saves the matrices as .txt
clc
R1=diag([1,1,1]);
t1=[0 0 0]';
R2=stereoParams.RotationOfCamera2;
t2=stereoParams.TranslationOfCamera2';

% Adjust if wrong from calibration
% t2(1)=530;
%t2(2)=0;
 %t2(3)=0;

R2b=[R2;0 0 0];
R1b=[R1;0 0 0];
T1=[R1b [t1; 1]];
T2=T1*[R2b [t2; 1]];

extrinsic1=[R1(1,:) t1(1);R1(2,:) t1(2);R1(3,:) t1(3)];
extrinsic2=[R2(1,:) t2(1);R2(2,:) t2(2);R2(3,:) t2(3)];

K1=stereoParams.CameraParameters1.IntrinsicMatrix';
K2=stereoParams.CameraParameters2.IntrinsicMatrix';

P1=K1*extrinsic1;
P2=K2*extrinsic2;


save('lcm_vlh2.txt','P1','-ascii');
save('rcm_vlh2.txt','P2','-ascii');
%% Detect intersections between squares
clc
left=imread('images\Chessboard_images\leftcalibration07_16_21.jpg');
right=imread('images\Chessboard_images\rightcalibration07_16_21.jpg');
[left2,~] = undistortImage(left,stereoParams.CameraParameters1);
[right2,~] = undistortImage(right,stereoParams.CameraParameters2);

[imagePointsLeft,~] = detectCheckerboardPoints(left);
[imagePointsRight,~] = detectCheckerboardPoints(right);

% Undistort them
imagePointsLeft2 = undistortPoints(imagePointsLeft,stereoParams.CameraParameters1);
imagePointsRight2 = undistortPoints(imagePointsRight,stereoParams.CameraParameters2);
%%
save('lpoints3.txt','imagePointsLeft2','-ascii')
save('rpoints3.txt','imagePointsRight2','-ascii')

%% Plot the points
figure(1)
imshow(left);
hold on;
plot(imagePointsLeft2(:,1),imagePointsLeft2(:,2),'bo');
figure(2)
imshow(right);
hold on;
plot(imagePointsRight2(:,1),imagePointsRight2(:,2),'bo');


