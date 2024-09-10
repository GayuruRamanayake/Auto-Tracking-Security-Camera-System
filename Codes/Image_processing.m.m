% Clear all 
clc;
clear;
close all;

% object for foreground detection 
fgDetector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
%object for blob analysis
Analyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', true, 'CentroidOutputPort', true, ...
    'MinimumBlobArea', 400);

%Webcam object
camObj = VideoReader("intruder.mp4");

%video player object
framePrev = readFrame(camObj);
frameSize = size(framePrev);
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)] + 30]);

%create the people detector object
peopleDetector = peopleDetectorACF;

%video processing
while hasFrame(camObj)
    % Capture two frames
    frame1 = framePrev;
    %framePrev = snapshot(camObj); 
   framePrev     = readFrame(camObj);
    % Find the absolute difference between two frames
    frameDiff = imabsdiff(frame1, framePrev);

    % Convert the difference image to grayscale
    grayDiff = rgb2gray(frameDiff);

    % Apply Gaussian smoothing filter to the grayscale difference image
    blurredDiff = imgaussfilt(grayDiff, 'FilterSize', 5);

    % Detect foreground 
    mask = step(fgDetector, blurredDiff);

    % remove noise and fill in holes
    mask = imopen(mask, strel('rectangle', [3, 3]));
    mask = imclose(mask, strel('rectangle', [15, 15]));
    mask = imfill(mask, 'holes');

    % Perform blob analysis to find connected components
    [blobArea, blobCentroids, blobBoundingBoxes] = step(Analyser, mask);

    % Annotate the detected objects
    frame1 = insertObjectAnnotation(frame1, 'rectangle', blobBoundingBoxes, 1);

    % Check if the 'g' key is pressed to stop analysis
    figure(1);
    KeyPressed = ~isempty(get(gcf, 'CurrentCharacter'));
    if KeyPressed
        % Clean up and exit the loop if 'g' key is pressed
        clear camObj;
        release(videoPlayer);
        break;
    end
    
    %get detected boxes and corresponding scores from peopleDetectorACF
    %model
    [detectedBoxes, detectionScore] = detect(peopleDetector, frame1);
    %apply constraints for prediction scores
    if (sum(sum(detectedBoxes)) > 20)
        %annotate detected frames containing people in green with
        %prediction scores
        annotatedFrame = insertObjectAnnotation(frame1, 'rectangle', detectedBoxes, detectionScore, "Color", "green");
        %display each frame
        imshow(annotatedFrame);
        title('Possible intruder detected');
        %play alarm tone 
        [y, fs] = audioread('mixkit-alarm-tone-996.wav');
        sound(y, fs);
    else
        imshow(frame1);
        title('No People Detected');
    end
    
end

% Clean up resources
clear camObj;
release(videoPlayer);
release(fgDetector);
