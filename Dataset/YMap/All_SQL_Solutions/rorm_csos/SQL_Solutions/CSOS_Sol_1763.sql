-- CREATE DATABASE FOR /home/centos/orm_synthesizer/models/CSOS/ImplSolution/CSOS_Sol_1763.sql

USE CSOS_Sol_1763;
--
-- Table structure for table SMSChannel
--

CREATE TABLE `SMSChannel` (
`telNo` varchar(64),
`smsProvider` varchar(64),
`channelID` int NOT NULL, 
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
-- Table structure for table ProcessStateMachineEvent
--

CREATE TABLE `ProcessStateMachineEvent` (
`eventID` int NOT NULL, 
`stateMachineID` int,
KEY `FK_ProcessStateMachineEvent_stateMachineID_idx` (`stateMachineID`),
PRIMARY KEY (`eventID`)
);

--
-- Table structure for table Channel
--

CREATE TABLE `Channel` (
`channelID` int NOT NULL, 
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
-- Table structure for table Viewer
--

CREATE TABLE `Viewer` (
`period` int,
`principalID` int NOT NULL, 
KEY `FK_Viewer_principalID_idx` (`principalID`),
PRIMARY KEY (`principalID`)
);

--
-- Table structure for table PrincipalProxy
--

CREATE TABLE `PrincipalProxy` (
`principalID` int NOT NULL, 
`channelID` int NOT NULL, 
KEY `FK_PrincipalProxy_principalID_idx` (`principalID`),
KEY `FK_PrincipalProxy_channelID_idx` (`channelID`),
PRIMARY KEY (`principalID`,`channelID`)
);

--
-- Table structure for table Institution
--

CREATE TABLE `Institution` (
`InstitutionID` int,
`principalID` int NOT NULL, 
KEY `FK_Institution_principalID_idx` (`principalID`),
PRIMARY KEY (`principalID`)
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
-- Table structure for table Person
--

CREATE TABLE `Person` (
`principalName` varchar(64),
`role` int,
`principalID` int NOT NULL, 
PRIMARY KEY (`principalID`)
);

--
-- Table structure for table Principal
--

CREATE TABLE `Principal` (
`principalName` varchar(64),
`principalID` int NOT NULL, 
PRIMARY KEY (`principalID`)
);

ALTER TABLE `ProcessStateMachineEvent`
  ADD CONSTRAINT `FK_ProcessStateMachineEvent_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `ProcessStateMachineTransition`
  ADD CONSTRAINT `FK_ProcessStateMachineTransition_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `StateMachineStates`
  ADD CONSTRAINT `FK_StateMachineStates_stateID` FOREIGN KEY (`stateID`) REFERENCES `ProcessStateMachineState` (`stateID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_StateMachineStates_stateMachineID` FOREIGN KEY (`stateMachineID`) REFERENCES `ProcessStateMachine` (`stateMachineID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Viewer`
  ADD CONSTRAINT `FK_Viewer_principalID` FOREIGN KEY (`principalID`) REFERENCES `Principal` (`principalID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `PrincipalProxy`
  ADD CONSTRAINT `FK_PrincipalProxy_principalID` FOREIGN KEY (`principalID`) REFERENCES `Principal` (`principalID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_PrincipalProxy_channelID` FOREIGN KEY (`channelID`) REFERENCES `Channel` (`channelID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Institution`
  ADD CONSTRAINT `FK_Institution_principalID` FOREIGN KEY (`principalID`) REFERENCES `Principal` (`principalID`) ON DELETE CASCADE ON UPDATE CASCADE;

