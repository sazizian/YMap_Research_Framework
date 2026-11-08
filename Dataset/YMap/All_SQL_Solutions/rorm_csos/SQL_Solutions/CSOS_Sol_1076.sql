-- CREATE DATABASE FOR /home/centos/orm_synthesizer/models/CSOS/ImplSolution/CSOS_Sol_1076.sql

USE CSOS_Sol_1076;
--
-- Table structure for table SMSChannel
--

CREATE TABLE `SMSChannel` (
`telNo` varchar(64),
`smsProvider` varchar(64),
`channelID` int NOT NULL, 
KEY `FK_SMSChannel_channelID_idx` (`channelID`),
PRIMARY KEY (`channelID`)
);

--
-- Table structure for table ProcessStateMachine
--

CREATE TABLE `ProcessStateMachine` (
`stateMachineID` int NOT NULL, 
PRIMARY KEY (`stateMachineID`)
);

--
-- Table structure for table SecEmailChannel
--

CREATE TABLE `SecEmailChannel` (
`secEmailID` int,
`emailID` int,
`channelID` int NOT NULL, 
PRIMARY KEY (`channelID`)
);

--
-- Table structure for table ProcessStateMachineExecution
--

CREATE TABLE `ProcessStateMachineExecution` (
`currentStateID` int,
`processStateMachineID` int,
`processStateMachineExecutionID` int NOT NULL, 
PRIMARY KEY (`processStateMachineExecutionID`)
);

--
-- Table structure for table StateMachineEvents
--

CREATE TABLE `StateMachineEvents` (
`eventID` int NOT NULL, 
`stateMachineID` int NOT NULL, 
KEY `FK_StateMachineEvents_eventID_idx` (`eventID`),
KEY `FK_StateMachineEvents_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`eventID`,`stateMachineID`)
);

--
-- Table structure for table ProcessStateMachineEvent
--

CREATE TABLE `ProcessStateMachineEvent` (
`eventID` int NOT NULL, 
PRIMARY KEY (`eventID`)
);

--
-- Table structure for table Channel
--

CREATE TABLE `Channel` (
`principalID` int,
`channelID` int NOT NULL, 
KEY `FK_Channel_principalID_idx` (`principalID`),
PRIMARY KEY (`channelID`)
);

--
-- Table structure for table ProcessStateMachineTransition
--

CREATE TABLE `ProcessStateMachineTransition` (
`toStateID` int,
`fromStateID` int,
`transitionID` int NOT NULL, 
`stateMachineID` int,
KEY `FK_ProcessStateMachineTransition_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`transitionID`)
);

--
-- Table structure for table StateMachineStates
--

CREATE TABLE `StateMachineStates` (
`stateID` int NOT NULL, 
`stateMachineID` int NOT NULL, 
KEY `FK_StateMachineStates_stateID_idx` (`stateID`),
KEY `FK_StateMachineStates_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`stateID`,`stateMachineID`)
);

--
-- Table structure for table EmailChannel
--

CREATE TABLE `EmailChannel` (
`emailID` int,
`channelID` int NOT NULL, 
PRIMARY KEY (`channelID`)
);

--
-- Table structure for table ProcessStateMachineAction
--

CREATE TABLE `ProcessStateMachineAction` (
`actionStateMachineID` int,
`actionID` int NOT NULL, 
PRIMARY KEY (`actionID`)
);

--
-- Table structure for table ProcessStateMachineState
--

CREATE TABLE `ProcessStateMachineState` (
`exitActionID` int,
`entryActionID` int,
`stateID` int NOT NULL, 
PRIMARY KEY (`stateID`)
);

--
-- Table structure for table Principal
--

CREATE TABLE `Principal` (
`DType` varchar(64),
`principalName` varchar(64),
`InstitutionID` int,
`period` int,
`role` int,
`principalID` int NOT NULL, 
PRIMARY KEY (`principalID`)
);

ALTER TABLE `SMSChannel`
  ADD CONSTRAINT `FK_SMSChannel_channelID` FOREIGN KEY (`channelID`) REFERENCES `Channel` (`channelID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `StateMachineEvents`
  ADD CONSTRAINT `FK_StateMachineEvents_eventID` FOREIGN KEY (`eventID`) REFERENCES `ProcessStateMachineEvent` (`eventID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_StateMachineEvents_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Channel`
  ADD CONSTRAINT `FK_Channel_principalID` FOREIGN KEY (`principalID`) REFERENCES `Principal` (`principalID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `ProcessStateMachineTransition`
  ADD CONSTRAINT `FK_ProcessStateMachineTransition_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `StateMachineStates`
  ADD CONSTRAINT `FK_StateMachineStates_stateID` FOREIGN KEY (`stateID`) REFERENCES `ProcessStateMachineState` (`stateID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_StateMachineStates_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

