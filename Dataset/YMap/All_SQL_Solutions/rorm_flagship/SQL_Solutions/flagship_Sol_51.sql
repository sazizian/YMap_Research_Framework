-- CREATE DATABASE FOR /home/centos/orm_synthesizer/models/flagship_re/ImplSolution/flagship_re_Sol_51.sql

USE flagship_re_Sol_51;
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
KEY `FK_Category_groupID_idx` (`groupID`),
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
-- Table structure for table CategoryUserAssociation
--

CREATE TABLE `CategoryUserAssociation` (
`categoryID` int NOT NULL, 
`userID` int NOT NULL, 
KEY `FK_CategoryUserAssociation_categoryID_idx` (`categoryID`),
KEY `FK_CategoryUserAssociation_userID_idx` (`userID`),
PRIMARY KEY (`categoryID`,`userID`)
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
-- Table structure for table RevisionUserAssociation
--

CREATE TABLE `RevisionUserAssociation` (
`revisionID` int NOT NULL, 
`userID` int NOT NULL, 
KEY `FK_RevisionUserAssociation_revisionID_idx` (`revisionID`),
KEY `FK_RevisionUserAssociation_userID_idx` (`userID`),
PRIMARY KEY (`revisionID`,`userID`)
);

--
-- Table structure for table Revision
--

CREATE TABLE `Revision` (
`revisionName` varchar(64),
`revisionID` int NOT NULL, 
`documentID` int,
KEY `FK_Revision_documentID_idx` (`documentID`),
PRIMARY KEY (`revisionID`)
);

--
-- Table structure for table CategoryBackgroundAssociation
--

CREATE TABLE `CategoryBackgroundAssociation` (
`categoryID` int NOT NULL, 
`backgroundID` int NOT NULL, 
KEY `FK_CategoryBackgroundAssociation_categoryID_idx` (`categoryID`),
KEY `FK_CategoryBackgroundAssociation_backgroundID_idx` (`backgroundID`),
PRIMARY KEY (`categoryID`,`backgroundID`)
);

ALTER TABLE `Category`
  ADD CONSTRAINT `FK_Category_groupID` FOREIGN KEY (`groupID`) REFERENCES `Group` (`groupID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `GroupUserAssociation`
  ADD CONSTRAINT `FK_GroupUserAssociation_groupID` FOREIGN KEY (`groupID`) REFERENCES `Group` (`groupID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_GroupUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `CategoryUserAssociation`
  ADD CONSTRAINT `FK_CategoryUserAssociation_categoryID` FOREIGN KEY (`categoryID`) REFERENCES `Category` (`categoryID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_CategoryUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `DocumentUserAssociation`
  ADD CONSTRAINT `FK_DocumentUserAssociation_documentID` FOREIGN KEY (`documentID`) REFERENCES `Document` (`documentID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_DocumentUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Document`
  ADD CONSTRAINT `FK_Document_categoryID` FOREIGN KEY (`categoryID`) REFERENCES `Category` (`categoryID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `RevisionUserAssociation`
  ADD CONSTRAINT `FK_RevisionUserAssociation_revisionID` FOREIGN KEY (`revisionID`) REFERENCES `Revision` (`revisionID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_RevisionUserAssociation_userID` FOREIGN KEY (`userID`) REFERENCES `User` (`userID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `Revision`
  ADD CONSTRAINT `FK_Revision_documentID` FOREIGN KEY (`documentID`) REFERENCES `Document` (`documentID`) ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE `CategoryBackgroundAssociation`
  ADD CONSTRAINT `FK_CategoryBackgroundAssociation_categoryID` FOREIGN KEY (`categoryID`) REFERENCES `Category` (`categoryID`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `FK_CategoryBackgroundAssociation_backgroundID` FOREIGN KEY (`backgroundID`) REFERENCES `Background` (`backgroundID`) ON DELETE CASCADE ON UPDATE CASCADE;

