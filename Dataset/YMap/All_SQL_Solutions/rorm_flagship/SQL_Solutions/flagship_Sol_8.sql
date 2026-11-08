-- CREATE DATABASE FOR /home/centos/orm_synthesizer/models/flagship_re/ImplSolution/flagship_re_Sol_8.sql

USE flagship_re_Sol_8;
--
-- Table structure for table Group
--

CREATE TABLE `Group` (
`groupName` varchar(64),
`groupID` int NOT NULL, 
PRIMARY KEY (`groupID`)
);

--
-- Table structure for table User
--

CREATE TABLE `User` (
`userName` varchar(64),
`userID` int NOT NULL, 
PRIMARY KEY (`userID`)
);

--
-- Table structure for table Category
--

CREATE TABLE `Category` (
`categoryName` varchar(64),
`groupID` int,
`categoryID` int NOT NULL, 
`backgroundID` int,
`userID` int,
KEY `FK_Category_groupID_idx` (`groupID`),
KEY `FK_Category_backgroundID_idx` (`backgroundID`),
KEY `FK_Category_userID_idx` (`userID`),
PRIMARY KEY (`categoryID`)
);

--
-- Table structure for table Background
--

CREATE TABLE `Background` (
`backgroundName` varchar(64),
`backgroundID` int NOT NULL, 
PRIMARY KEY (`backgroundID`)
);

--
-- Table structure for table GroupUserAssociation
--

CREATE TABLE `GroupUserAssociation` (
`groupID` int NOT NULL, 
`userID` int NOT NULL, 
KEY `FK_GroupUserAssociation_groupID_idx` (`groupID`),
KEY `FK_GroupUserAssociation_userID_idx` (`userID`),
PRIMARY KEY (`groupID`,`userID`)
);

--
-- Table structure for table DocumentUserAssociation
--

CREATE TABLE `DocumentUserAssociation` (
`documentID` int NOT NULL, 
`userID` int NOT NULL, 
KEY `FK_DocumentUserAssociation_documentID_idx` (`documentID`),
KEY `FK_DocumentUserAssociation_userID_idx` (`userID`),
PRIMARY KEY (`documentID`,`userID`)
);

--
-- Table structure for table Document
--

CREATE TABLE `Document` (
`title` varchar(64),
`categoryID` int,
`documentID` int NOT NULL, 
KEY `FK_Document_categoryID_idx` (`categoryID`),
PRIMARY KEY (`documentID`)
);

--
-- Table structure for table Revision
--

CREATE TABLE `Revision` (
`revisionName` varchar(64),
`revisionID` int NOT NULL, 
`documentID` int,
`userID` int,
KEY `FK_Revision_documentID_idx` (`documentID`),
KEY `FK_Revision_userID_idx` (`userID`),
PRIMARY KEY (`revisionID`)
);

ALTER TABLE `Category`
  ADD CONSTRAINT `FK_Category_groupID` FOREIGN KEY (`groupID`) REFERENCES `Group` (`groupID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_Category_backgroundID` FOREIGN KEY (`backgroundID`) REFERENCES `Background` (`backgroundID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_Category_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `GroupUserAssociation`
  ADD CONSTRAINT `FK_GroupUserAssociation_groupID` FOREIGN KEY (`groupID`) REFERENCES `Group` (`groupID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_GroupUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `DocumentUserAssociation`
  ADD CONSTRAINT `FK_DocumentUserAssociation_documentID` FOREIGN KEY (`documentID`) REFERENCES `Document` (`documentID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_DocumentUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Document`
  ADD CONSTRAINT `FK_Document_categoryID` FOREIGN KEY (`categoryID`) REFERENCES `Category` (`categoryID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Revision`
  ADD CONSTRAINT `FK_Revision_documentID` FOREIGN KEY (`documentID`) REFERENCES `Document` (`documentID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_Revision_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

