import React from "react";
import { createStyles, makeStyles, Theme } from "@material-ui/core/styles";
import Button from "@material-ui/core/Button";

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    button: {
      margin: theme.spacing(1),
      color: "#00A7D9"
    },
    input: {
      display: "none"
    }
  })
);

interface Props {
  generateFace: Function;
}

export default function GeneratePictureButton({ generateFace }: Props) {
  const classes = useStyles();

  const handleOnClick = () => {
    generateFace();
  };

  return (
    <Button
      variant="outlined"
      color="primary"
      className={classes.button}
      onClick={handleOnClick}
    >
      Generate Face
    </Button>
  );
}
