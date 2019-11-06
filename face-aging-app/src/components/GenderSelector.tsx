import React, { ChangeEvent } from "react";

import { withStyles } from "@material-ui/core/styles";
import Radio, { RadioProps } from "@material-ui/core/Radio";
import { RadioGroup, FormControlLabel } from "@material-ui/core";

interface Props {
  label: string;
  targetGender: string;
  registerTargetGender: Function;
}

const BlueRadio = withStyles({
  root: {
    "&$checked": {
      color: "#00A7D9"
    }
  },
  checked: {}
})((props: RadioProps) => <Radio color="default" {...props} />);

export default function GenderSelector({
  label,
  registerTargetGender,
  targetGender
}: Props) {
  const handleChange = (event: ChangeEvent<unknown>) => {
    let value = (event.target as HTMLInputElement).value;
    console.log(value);
    registerTargetGender(value);
  };

  return (
    <div>
      <div>{label}</div>
      <RadioGroup
        row
        aria-label="radio-buttons"
        name="ageSelect"
        value={targetGender}
        onChange={handleChange}
      >
        <FormControlLabel
          value={"male"}
          control={<BlueRadio />}
          label={"male"}
          labelPlacement="bottom"
        />
        <FormControlLabel
          value={"female"}
          control={<BlueRadio />}
          label={"female"}
          labelPlacement="bottom"
        />
      </RadioGroup>
    </div>
  );
}
