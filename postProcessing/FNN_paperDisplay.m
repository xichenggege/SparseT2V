% Post processing for sparse T2V
% Evaluate reconstruction performance
clc
clear all
close all

% prepare folders
addpath('.\Functions');
mkdir(strcat('Figures\FNN_paperDisplay'));
dataLoc   = num2str(pwd);

%% Load reference data
folder_name = ['postProcessing'];
case_name   = ['PlaneJet2D_group1_1300cases';...
               'PlaneJet2D_group2_1120cases';...
               'PlaneJet2D_group3_1320cases'];

numSnap      = 1300+1120+1320;    % number of snapshots
numDimension = 95*160;  % number of dimensions

U = zeros(numDimension,numSnap);
T = zeros(numDimension,numSnap);

% Load all three groups
numCase = 1;
for k = 1:numel(case_name(:,1)) 
    file_name   = sprintf('%sTrainData\\%s.mat',...
        dataLoc(1:end-numel(folder_name)),case_name(k,:));
    fdata = load(file_name);
    
    for j = 1:numel(fdata.U)
        U(:,numCase) = fdata.U{j}(:);
        T(:,numCase) = fdata.T{j}(:);
        numCase = numCase + 1;
    end  
end

% coordinates, x and y
xmesh = fdata.xmesh;
ymesh = fdata.ymesh;
Xind  = xmesh(:);
Yind  = ymesh(:);

% Load BC
file_name   = sprintf('%sTrainData\\TrainData_BC.mat',...
        dataLoc(1:end-numel(folder_name)));
fdata = load(file_name);

BoundaryConditions = struct('U0',fdata.U_BC,'T0',fdata.T_BC,...
    'I0',fdata.I_BC,'mu',fdata.mu_BC,'Ieff',fdata.Ieff_BC);

%% data post-processing
% -------------------------------------------------------------------------
% _*Framework FDD 1: 2 MLPs to map TS-TL-UL* _ 
%   _case1: use optimal sensors from T field
%   _case2: use optimal sensors from both T and U field
%   _case3: use TCsw with similar arrangement as PPOOLEX test
% -------------------------------------------------------------------------
% _*Framework FDD 2: 1 MLPs to map TS-UL* _ 
%   _case1: use optimal sensors from T field
%   _case2: use optimal sensors from both T and U field
%   _case3: use TCsw with similar arrangement as PPOOLEX test
% -------------------------------------------------------------------------

case_name   = [{'FDD-1\_case1'};{'FDD-1\_case2'};{'FDD-1\_case3'};...
               {'FDD-2\_case1'};{'FDD-2\_case2'};{'FDD-2\_case3'}];

field_name  = [{'TS2UF'};{'TL2UL'};{'TS2TL'};{'TS2TF'}]; % For FDD-2, TL2UL -> TS2UL

num_run = 10; % run 10 times for each case
% Load prediction data
case_ = struct();
for k =1:numel(case_name) % each case
    
    for nfield = 1:numel(field_name) % number of field
        fdata = [];
        try
            for nRun = 1:num_run % number of run
                file_name = sprintf('%s%s\\run%d\\pred\\%s.mat',...
                dataLoc(1:end-numel(folder_name)),case_name{k},nRun-1,field_name{nfield});
        
                load(file_name);
                fdata.trainloss{nRun}     = hist_trainloss;
                fdata.valoss{nRun}        = hist_valoss;
                fdata.predict{nRun}       = predict;
                fdata.reference{nRun}     = reference;
                fdata.caseIndex{nRun}     = caseIndex;

                % Calculation of NMSE of each test case
                nmse = [];
                for i = 1:numel(caseIndex)
                    nmse(i) = NMSE(reference(:,i),predict(:,i));  % NMSE: reference - prediction
                end
                
                fdata.nmse{nRun}      = nmse;
                fdata.nmse_mean{nRun} = mean(nmse);

            end
        catch
            warning(sprintf('%s is not loaded in case named: %s',...
                field_name{nfield},case_name{k}));
        end
        case_(k).(field_name{nfield}) = fdata;
    end
end



%% Visualization
plotParameters;
lgd_name = [{'FDD-1-case1'};{'FDD-1-case2'};{'FDD-1-case3'};...
               {'FDD-2-case1'};{'FDD-2-case2'};{'FDD-2-case3'}];
set(0,'DefaultFigureVisible', 'on');  % avoid memory explosion 
%--------------------------------------------------------------------------
% Mean NMSE error between cases and  each run 'TS2UF'
%--------------------------------------------------------------------------

FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);

for k = 1:numel(case_name)
    yp = cell2mat(case_(k).TS2UF.nmse_mean);
    plot(yp)
end
legend(lgd_name)

%--------------------------------------------------------------------------
% Train history
%--------------------------------------------------------------------------
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);

k=3; nRun = 9;
yp = case_(k).TS2UF.trainloss{nRun};
plot(yp)
yp = case_(k).TS2UF.valoss{nRun};
plot(yp)
set(gca, 'YScale', 'log')
legend('train','val')


%--------------------------------------------------------------------------
% Mean NMSE error between cases and runs 'TS2UF'
%--------------------------------------------------------------------------
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);

% TS2UF
for k = 1:numel(case_name)
    yp(k) = mean(cell2mat(case_(k).TS2UF.nmse_mean));
    yp_std(k) = std(cell2mat(case_(k).TS2UF.nmse_mean));
end
% group mean and std
yp     = [yp([1,4]);yp([2,5]);yp([3,6])];
yp_std = [yp_std([1,4]);yp_std([2,5]);yp_std([3,6])];

% TS2TF
for k = 1:3
    yp(k,3) = mean(cell2mat(case_(k).TS2TF.nmse_mean));
    yp_std(k,3) = std(cell2mat(case_(k).TS2TF.nmse_mean));
end
% bar plot
bp     = bar(yp);
set(gca, 'YScale', 'log')

% to enable add error bar over each bar plot
[ngroups,nbars] = size(yp);
% Get the x coordinate of the bars
xc = nan(nbars, ngroups);
for i = 1:nbars
    xc(i,:) = bp(i).XEndPoints;
end
% add errorbar over bar
errorbar(xc',yp,yp_std,'k','linestyle','none','lineWidth',2);
set(gca,'ytick',[10e-4 10e-3 10e-2 10e-1 10e0]);
set(gca,'ylim',[1e-4 2e1]);
% figure properties
x_label_name = {'Case1','Case2','Case3'};
set(gca, 'XTick', 1:length(x_label_name),'XTickLabel',x_label_name);
ylabel('NMSE [-]','interpreter','latex','fontsize',18,'fontname','times');
set(gca,'fontsize',18,'fontname','times'); 
legend('FDD-1: TS2UF','FDD-2: TS2UF','FDD-1: TS2TF',...
        'NumColumns',2,'location','northeast','fontsize',15);
legend('boxoff')
exportgraphics(FigHandle,['Figures\FNN_paperDisplay\' ...
    'NMSE_case_comparsion_TS2UF_TS2TF.png'],'Resolution',450);

%--------------------------------------------------------------------------
% Mean NMSE error between cases and runs 'TL2UL'
%--------------------------------------------------------------------------
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);

for k = 1:numel(case_name)
    yp(k) = mean(cell2mat(case_(k).TL2UL.nmse_mean));
    yp_std(k) = std(cell2mat(case_(k).TL2UL.nmse_mean));
end
% group mean and std
yp     = [yp([1,4]);yp([2,5]);yp([3,6])];
yp_std = [yp_std([1,4]);yp_std([2,5]);yp_std([3,6])];
% bar plot
bp     = bar(yp);

% to enable add error bar over each bar plot
[ngroups,nbars] = size(yp);
% Get the x coordinate of the bars
xc = nan(nbars, ngroups);
for i = 1:nbars
    xc(i,:) = bp(i).XEndPoints;
end
% add errorbar over bar
errorbar(xc',yp,yp_std,'k','linestyle','none','lineWidth',2);

% figure properties
x_label_name = {'Case1','Case2','Case3'};
set(gca, 'XTick', 1:length(x_label_name),'XTickLabel',x_label_name);
legend('FDD-1','FDD-2','location','best');
legend('boxoff')
ylabel('NMSE [-]','interpreter','latex','fontsize',18,'fontname','times');

set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,['Figures\FNN_paperDisplay\' ...
    'NMSE_case_comparsion_TL2UL.png'],'Resolution',450);

%--------------------------------------------------------------------------
% Mean NMSE error between cases and runs 'TS2TL'
%--------------------------------------------------------------------------
FigHandle = figure; hold on
set(FigHandle, 'Position', [150, 80, 600, 450]);
yp = [];
yp_std = [];
for k = 1:3
    yp(k) = mean(cell2mat(case_(k).TS2TL.nmse_mean));
    yp_std(k) = std(cell2mat(case_(k).TS2TL.nmse_mean));
end
% group mean and std

% bar plot
bp     = bar(yp);

xc = 1:1:3;
% add errorbar over bar
errorbar(xc',yp,yp_std,'k','linestyle','none','lineWidth',2);

% figure properties
x_label_name = {'Case1','Case2','Case3'};
set(gca, 'XTick', 1:length(x_label_name),'XTickLabel',x_label_name);
legend('FDD-1');
legend('boxoff')
ylabel('NMSE [-]','interpreter','latex','fontsize',18,'fontname','times');

set(gca,'fontsize',18,'fontname','times'); 
exportgraphics(FigHandle,['Figures\FNN_paperDisplay\' ...
    'NMSE_case_comparsion_TS2TL.png'],'Resolution',450);


%--------------------------------------------------------------------------
% Coefficient comparsion 
%--------------------------------------------------------------------------
% Use case 6, run 
% TCs grid, run8 with minimum nmse

%% Remeber to revise the index for 'run' and 'case'
k      = 6;
nRun   = 8;
nModes = 5;  % number of modes to plot

% Save in a sub folder
savefolder = sprintf('Figures\\FNN_paperDisplay\\%s-run%d',lgd_name{k},nRun);
mkdir(savefolder);

nRun  = nRun +1;
ypred = case_(k).TL2UL.predict{nRun};
yref  = case_(k).TL2UL.reference{nRun};
xp    = 1:1:numel(case_(k).TL2UL.caseIndex{nRun});

for modeIndex = 1:nModes
    FigHandle = figure; hold on
    set(FigHandle, 'Position', [150, 80, 800, 400]);
    
    h(1)  = plot(xp,ypred(modeIndex,:),'-','color','r','linewidth',1); 
    h(2)  = plot(xp,yref(modeIndex,:),'--','color','b','linewidth',1.5); 

    % Legend
    legend({'Prediction';'True'},...
        'Location','best','interpreter','latex','fontsize',18,'fontname','times');
    legend('boxoff');
    title (sprintf('Mode %d',modeIndex),'interpreter','latex','fontsize',16,'fontname','times');
    set(gca,'xlim',[0 385]);
%     set(gca,'ylim',[-0.1 1.1]);
    xlabel('Case\ index','interpreter','latex');
    ylabel('Mode\ coefficient','interpreter','latex');
    set(gca,'fontsize',18,'fontname','times'); 
    exportgraphics(FigHandle,sprintf(['%s\\Mode_coeff_modes_%d.png'],...
        savefolder,modeIndex),'Resolution',450);
end

% close all

%--------------------------------------------------------------------------
% Errors distribution vs case index
%--------------------------------------------------------------------------
i = 1;
for k = [3 6]
    nmsei = [];
    for  nRun = 1:9
        nmsei(nRun,:) = case_(k).TS2UF.nmse{nRun};
    end
    nmse_{i} = mean(nmsei,1);   % average all run nmse
    i = i+1;
end

testIndex    = case_(k).TS2UF.caseIndex{nRun}+1;
fieldname_   = fieldnames(BoundaryConditions);

disc_edges  = {[0:1:10];[20:5:80];[5 10:10:70]; [10 500 1000:1000:5000]; [0.1:0.1:3.0]};
xlabel_name = {'$U_0$ [m/s]';'$T_0\ [^\circ C]$';'$I_0$ [\%]';'$\mu_l/\mu_t$ [-]';'$I_{eff}$ [-]'};

disc_ = [];
for varIndex = 1:length(fieldname_)

    % discretize erros into edges to analyze the error source
    X              = BoundaryConditions.(fieldname_{varIndex})(testIndex) ;      % data to be discretized

    disc_{1}        = discretizeBC(X,disc_edges{varIndex},nmse_{1});  % FDD-1-Case3
    disc_{2}        = discretizeBC(X,disc_edges{varIndex},nmse_{2});  % FDD-2-Case3

    FigHandle = figure; hold on
    set(FigHandle, 'Position', [150, 80, 600, 450]);
    
    yp     = [disc_{1}.nmse_mean; disc_{2}.nmse_mean]';
    yp_std = [disc_{1}.nmse_std; disc_{2}.nmse_std]';

    % bar plot
     bp     = bar(yp);
    set(gca, 'YScale', 'log')
    
    % to enable add error bar over each bar plot
    [ngroups,nbars] = size(yp);
    % Get the x coordinate of the bars
    xc = nan(nbars, ngroups);
    for i = 1:nbars
        xc(i,:) = bp(i).XEndPoints;
    end
    % add errorbar over bar
    errorbar(xc',yp,yp_std,'k','linestyle','none','lineWidth',2);
    
    legend('FDD-1-Case3','FDD-2-Case3','location','best');
    % set(gca,'ytick',[10e-4 10e-3 10e-2 10e-1 10e0]);
    % set(gca,'ylim',[1e-4 2e1]);
    % figure properties 
    for j = 1:numel(disc_edges{varIndex})-1
        x_label_name{j} = num2str(0.5*(disc_edges{varIndex}(j)+disc_edges{varIndex}(j+1)));
    end
    set(gca, 'XTick', 1:length(x_label_name),'XTickLabel',x_label_name);
    xlabel(xlabel_name{varIndex},'interpreter','latex','fontsize',18,'fontname','times');
    ylabel('NMSE [-]','interpreter','latex','fontsize',18,'fontname','times');
    set(gca,'fontsize',18,'fontname','times'); 
     
    exportgraphics(FigHandle,sprintf('%s\\NMSE_error_source_distribution_log_BC_%s.png', ...
        savefolder,fieldname_{varIndex}),'Resolution',450);
end
close all

%--------------------------------------------------------------------------
% Contours comparsion, ref, pred, ref-pred
%--------------------------------------------------------------------------
caseIndex = 6;
nRun      = 8;
nRun      = nRun +1;
d         = 16e-3;
displayIndex = 1:1:numel(testIndex);   % bad, ok, good results index
% cmax_error = [0.2 0.02 1e-3 0.015 0.08];
set(0,'DefaultFigureVisible', 'off');  % avoid memory explosion 
for i = displayIndex
    case_lgd = [{'$ref\ U$'};{'$pred$'};{'$|ref-pred|$'}];
    
    % field plot
    FigHandle = figure; hold on
    set(FigHandle, 'Position', [150, 80, 1320, 340]);
    t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');    
    
    ref_  =reshape(case_(caseIndex).TS2UF.reference{nRun}(:,i),size(xmesh'))';
    pred_ =reshape(case_(caseIndex).TS2UF.predict{nRun}(:,i),size(xmesh'))';

    for plotIndex = 1:3 % ref, case1, case2, case3, ref-case3
        if ismember(plotIndex,[1])
            field =  ref_;
            cmin  = min(field(:));
            cmax  = max(field(:));
        elseif ismember(plotIndex,[2])
            field = pred_;
        elseif ismember(plotIndex,[3])
            field = abs(ref_-pred_);
            cmax  = max(field(:));
        end
    
        nexttile 
        
        pcolor(xmesh/d,ymesh/d,field)
        set(gca,'xlim',[0 125]);
        set(gca,'ylim',[-30 30]);
    
        shading interp;  
        colormap jet;
    
        if plotIndex == 1
            ylabel('$x/d$','interpreter','latex');
            xlabel('$y/d$','interpreter','latex');
            set(gca,'xtick',[0:25:125]);
            set(gca,'ytick',[-30:15:30]);
        else
            set(gca,'xtick',[],'ytick',[]);
        end
    
        caxis([0 0.85*cmax]); % for better display
    
        title (case_lgd{k},'interpreter','latex','fontsize',15,'fontname','times');
        set(gca,'fontsize',16,'fontname','times'); 
    
        if ismember(k,[1,3])
            colorbar;
        end
        axis equal
    end

    title_name = sprintf(['Case%d: $U_0$=%.1fm/s, $T_0$=%.1f$^\\circ$C, I=%.0f\\%%, $\\mu_{t}/\\mu_{l}$=%d, $I_{eff}$=%.1f \\%%'],...
    testIndex(i), BoundaryConditions.('U0')(testIndex(i)), BoundaryConditions.('T0')(testIndex(i)),...
    BoundaryConditions.('I0')(testIndex(i)), BoundaryConditions.('mu')(testIndex(i)),BoundaryConditions.('Ieff')(testIndex(i)));
    sgtitle(title_name,'fontsize',16,'fontname','Times','interpreter','latex');
    
    exportgraphics(FigHandle,sprintf('%s\\Ucontour_ref_pred_testCase_%d.png', ...
        savefolder,testIndex(i)),'Resolution',450);
    close all
end
