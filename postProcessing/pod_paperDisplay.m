% Post processing for sparse T2V
% Evaluate reconstruction performance
clc
clear all
close all

% prepare folders
addpath('.\Functions');
mkdir(strcat('Figures\pod_paperDisplay'));
dataLoc   = num2str(pwd);

%% Load reference data
folder_name = ['postProcessing'];
case_name   = ['PlaneJet2D_group1_1300cases'];

numSnap     = 1300;    % number of snapshots

file_name   = sprintf('%sTrainData\\%s.mat',...
    dataLoc(1:end-numel(folder_name)),case_name(1,:));
fdata = load(file_name);
   
% coordinates, x and y
xmesh = fdata.xmesh;
ymesh = fdata.ymesh;

%% data post-processing
%  extract POD components of U and T
%  and eigen value S
%  and optimal sensor placement

% Load prediction data
case_name   = 'FDD-1\pod';

field_name  = [{'TF2TL'};{'UF2UL'};]; 

POD = struct();
for nfield = 1:numel(field_name) % number of field
    file_name   = sprintf('%s%s\\pred\\%s.mat',...
        dataLoc(1:end-numel(folder_name)),case_name,field_name{nfield});
    POD.(field_name{nfield}) = load(file_name);

    % Calculation of NMSE of each test case
    nmse = [];
    for i = 1:numel(POD.(field_name{nfield}).prediction(1,:))
        nmse(i) = NMSE(POD.(field_name{nfield}).reference(:,i),...
                       POD.(field_name{nfield}).prediction(:,i));  % NMSE: reference - prediction
    end
    POD.(field_name{nfield}).nmse      = nmse;
    POD.(field_name{nfield}).nmse_mean = mean(nmse);
end

%% Visualization
plotParameters
%--------------------------------------------------------------------------
% Reconverd variance of number of modes
%--------------------------------------------------------------------------
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);

lgd_name = [{'T'};{'U'};]; 
for nfield = 1:numel(field_name)
    xp = 1:1:numel(POD.(field_name{nfield}).explained_variance_ratio_);
    yp = cumsum(POD.(field_name{nfield}).explained_variance_ratio_);
    f1(nfield)=plot(xp,100*yp,'-','color',co(nfield,:),'linewidth',1.5); 
    % 99.9% variance
    mode_threshold = min(find(yp>=0.999));
    plot([mode_threshold,mode_threshold],[0,101],'--','color',co(nfield,:),'linewidth',1.5); 
end
text(10.774193548387098,97.85673352435532,'99.9% variance','fontsize',18,'fontname','times');

% Legend
legend(f1,lgd_name,'Location','best','interpreter','latex','fontsize',18,'fontname','times');
legend('boxoff');

set(gca,'xlim',[0 18]);
set(gca,'ylim',[80 102]);

xlabel('Number of modes','interpreter','latex');
ylabel('Recostructed varaince (%)','interpreter','latex');
set(gca,'fontsize',18,'fontname','times'); 

exportgraphics(FigHandle,sprintf(['Figures\\pod_paperDisplay\\' ...
        'modes_recon_variance.png']),'Resolution',450);
exportgraphics(FigHandle,sprintf(['Figures\\pod_paperDisplay\\' ...
        'modes_recon_variance.emf']),'Resolution',450);

%--------------------------------------------------------------------------
% Plot first 2 modes 
%--------------------------------------------------------------------------
numModes = 5;
d        = 16e-3;  % nozzle diamter

% color bar jet
for modeIndex = 1:numModes % plot first 10 modes
    for nfield = 1:numel(field_name)
        FigHandle = figure; hold on
        set(FigHandle, 'Position', [150, 80, 600, 380]);
    
        fieldplot_ = squeeze(POD.(field_name{nfield}).pcaComponents(:,:,modeIndex))';
        cmin = min(fieldplot_(:)); 
        cmax = max(fieldplot_(:)); 
%         contourFactor = 0.5;
%         makePlotNoLine(xmesh/d,ymesh/d,fieldplot_,contourFactor);
        pcolor(xmesh/d,ymesh/d,fieldplot_); shading interp
        colormap jet;
%         colorbar
        axis equal
        caxis([0.85*cmin 0.85*cmax]); % for better display

        ylabel('$x/d$','interpreter','latex');
        xlabel('$y/d$','interpreter','latex');
        set(gca,'xtick',[0:25:125]);
        set(gca,'ytick',[-30:15:30]);
    
        title (sprintf('Mode %d',modeIndex),'interpreter','latex','fontsize',16,'fontname','times');
        set(gca,'fontsize',16,'fontname','times'); 

        set(gca,'xlim',[0 125]);
        set(gca,'ylim',[-30 30]);

        exportgraphics(FigHandle,sprintf(['Figures\\pod_paperDisplay\\' ...
        'pod_%s_mode_%d.png'],lgd_name{nfield},modeIndex),'Resolution',900);
            exportgraphics(FigHandle,sprintf(['Figures\\pod_paperDisplay\\' ...
        'pod_%s_mode_%d.emf'],lgd_name{nfield},modeIndex),'Resolution',450);
    end
end

% Colorbar redblue
for modeIndex = 1:numModes % plot first 10 modes
    for nfield = 1:numel(field_name)
        FigHandle = figure; hold on
        set(FigHandle, 'Position', [150, 80, 600, 350]);
    
        fieldplot_ = squeeze(POD.(field_name{nfield}).pcaComponents(:,:,modeIndex))';

        contourFactor = 0.5;
        makePlotNoLine(xmesh/d,ymesh/d,fieldplot_,contourFactor);
      
        ylabel('$x/d$','interpreter','latex');
        xlabel('$y/d$','interpreter','latex');
        set(gca,'xtick',[0:25:125]);
        set(gca,'ytick',[-30:15:30]);

        title (sprintf('Mode %d',modeIndex),'interpreter','latex','fontsize',16,'fontname','times');
        set(gca,'fontsize',16,'fontname','times'); 

        set(gca,'xlim',[0 125]);
        set(gca,'ylim',[-30 30]);

        exportgraphics(FigHandle,sprintf(['Figures\\pod_paperDisplay\\' ...
        'pod_%s_mode_%d_redblue.png'],lgd_name{nfield},modeIndex),'Resolution',450);
    end
end

% close all

%% sensor position plot
%-----------------------------------
% Sensors position and contour of T
%-----------------------------------
displayIndex = 210;
ref          = fdata.T{displayIndex};


% sensor index of _case 1
sensorsIndexT     = [ 8160,   144,  8487,  7552,  6884,  9145,  4221,  5463,15180, 6726];
% sensor index of _case 2
sensorsIndexU     = [ 7552, 7522, 4707, 5633, 6725];
% sensor index of _case 3
sensorsIndex_grid = [ 4803,  4810,  4815,  4830,  4845,  4860,  4890,  4940,  6403,...
        6410,  6415,  6430,  6445,  6460,  6490,  6540,  7523,  7530,...
        7535,  7550,  7565,  7580,  7610,  7660,  8643,  8650,  8655,...
        8670,  8685,  8700,  8730,  8780, 10243, 10250, 10255, 10270,...
       10285, 10300, 10330, 10380];

Xind  = xmesh';
Xind  = Xind(:);
Yind  = ymesh';
Yind  = Yind(:);


FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 400]);
cmax = max(ref(:));
cmin = min(ref(:));

pcolor(xmesh/d,ymesh/d,ref)
plot(Xind(sensorsIndexT)/d,Yind(sensorsIndexT)/d,'o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',6) % sensor location
plot(Xind(sensorsIndexU)/d,Yind(sensorsIndexU)/d,'o','MarkerEdgeColor','w',...
    'MarkerFaceColor','w','MarkerSize',6) % sensor location
% 
% plot(Xind(sensorsIndex_grid)/d,Yind(sensorsIndex_grid)/d,'d','MarkerEdgeColor','r',...
%     'MarkerFaceColor','r','MarkerSize',6) % sensor location

% caxis([0.8*cmin 0.8*cmax]); 
colormap jet
% colorbar;
shading interp;  
axis equal 

xlabel('$x/d$','interpreter','latex','fontsize',18,'fontname','times');
ylabel('$y/d$','interpreter','latex','fontsize',18,'fontname','times');
set(gca,'ytick',[-50:25:50]);
set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,strcat('Figures\pod_paperDisplay\SensorPosition_1.png'),'Resolution',450);
exportgraphics(FigHandle,strcat('Figures\pod_paperDisplay\SensorPosition_1.emf'),'Resolution',450);

FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 400]);
cmax = max(ref(:));
cmin = min(ref(:));

pcolor(xmesh/d,ymesh/d,ref)
% 
plot(Xind(sensorsIndex_grid)/d,Yind(sensorsIndex_grid)/d,'o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',6) % sensor location

% caxis([0 0.8*cmax]); 
colormap jet
% colorbar;
shading interp;  
axis equal 

xlabel('$x/d$','interpreter','latex','fontsize',18,'fontname','times');
ylabel('$y/d$','interpreter','latex','fontsize',18,'fontname','times');
set(gca,'ytick',[-50:25:50]);
set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,strcat('Figures\pod_paperDisplay\SensorPosition_2.png'),'Resolution',450);
exportgraphics(FigHandle,strcat('Figures\pod_paperDisplay\SensorPosition_2.emf'),'Resolution',450);
