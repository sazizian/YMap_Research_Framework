
USE CSOS_Sol_1;


CREATE TABLE `SMSChannel` (
`telNo` varchar(64),
`smsProvider` varchar(64),
`channelID` int NOT NULL, 
KEY `FK_SMSChannel_channelID_idx` (`channelID`),
PRIMARY KEY (`channelID`)
);



CREATE TABLE `ProcessStateMachine` (
`stateMachineID` int NOT NULL, 
PRIMARY KEY (`stateMachineID`)
);



CREATE TABLE `SecEmailChannel` (
`secEmailID` int,
`channelID` int NOT NULL, 
KEY `FK_SecEmailChannel_channelID_idx` (`channelID`),
PRIMARY KEY (`channelID`)
);



CREATE TABLE `ProcessStateMachineExecution` (
`currentStateID` int,
`processStateMachineID` int,
`processStateMachineExecutionID` int NOT NULL, 
PRIMARY KEY (`processStateMachineExecutionID`)
);


CREATE TABLE `ProcessStateMachineEvent` (
`eventID` int NOT NULL, 
`stateMachineID` int,
KEY `FK_ProcessStateMachineEvent_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`eventID`)
);


CREATE TABLE `Channel` (
`principalID` int,
`channelID` int NOT NULL, 
KEY `FK_Channel_principalID_idx` (`principalID`),
PRIMARY KEY (`channelID`)
);



CREATE TABLE `ProcessStateMachineTransition` (
`toStateID` int,
`fromStateID` int,
`transitionID` int NOT NULL, 
`stateMachineID` int,
KEY `FK_ProcessStateMachineTransition_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`transitionID`)
);



CREATE TABLE `Viewer` (
`period` int,
`principalID` int NOT NULL, 
KEY `FK_Viewer_principalID_idx` (`principalID`),
PRIMARY KEY (`principalID`)
);


CREATE TABLE `Institution` (
`InstitutionID` int,
`principalID` int NOT NULL, 
KEY `FK_Institution_principalID_idx` (`principalID`),
PRIMARY KEY (`principalID`)
);


CREATE TABLE `EmailChannel` (
`emailID` int,
`channelID` int NOT NULL, 
PRIMARY KEY (`channelID`)
);



CREATE TABLE `ProcessStateMachineAction` (
`actionStateMachineID` int,
`actionID` int NOT NULL, 
PRIMARY KEY (`actionID`)
);

CREATE TABLE `ProcessStateMachineState` (
`exitActionID` int,
`entryActionID` int,
`stateID` int NOT NULL, 
`stateMachineID` int,
KEY `FK_ProcessStateMachineState_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`stateID`)
);



CREATE TABLE `Person` (
`role` int,
`principalID` int NOT NULL, 
KEY `FK_Person_principalID_idx` (`principalID`),
PRIMARY KEY (`principalID`)
);



CREATE TABLE `Principal` (
`principalName` varchar(64),
`principalID` int NOT NULL, 
PRIMARY KEY (`principalID`)
);

